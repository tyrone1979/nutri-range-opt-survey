#!/usr/bin/env python3
"""Process-pool helpers for fair-budget algorithm runs (shared by E1/E2/E3)."""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from benchmark_core import RunBudget, append_log

# Per-process state (set by initializer; avoids re-pickling FoodSpace per job)
_WORKER_FS = None
_WORKER_BUDGET: Optional[RunBudget] = None


def default_workers(cli: Optional[int] = None, cfg: Optional[dict] = None) -> int:
    if cli is not None:
        return max(1, int(cli))
    if cfg and cfg.get("workers") is not None:
        return max(1, int(cfg["workers"]))
    env = os.environ.get("NUTRIOPT_WORKERS")
    if env:
        return max(1, int(env))
    n = os.cpu_count() or 4
    # Leave a couple cores free; cap to avoid RAM blow-up on huge matrices
    return max(1, min(32, n - 2))


def _init_worker(fs, budget: RunBudget) -> None:
    global _WORKER_FS, _WORKER_BUDGET
    _WORKER_FS = fs
    _WORKER_BUDGET = budget


def _empty_res(budget: RunBudget) -> dict:
    return {
        "ffg": budget.n_generations + 1,
        "ffg_evals": budget.max_evaluations,
        "fhr": 0.0,
        "fsc": 0,
        "best_fitness": 0.0,
        "n_evals": 0,
        "diversity": 0.0,
        "avg_n_foods": 0.0,
        "contrib": 0.0,
    }


def algo_worker_job(job: dict) -> dict:
    """
    Execute one (algorithm, target, seed) run in a worker.

    job keys:
      algorithm, seed, target (Target), meta (dict merged into output row)
    """
    from run_benchmark import run_algorithm  # late import for spawn/fork safety

    global _WORKER_FS, _WORKER_BUDGET
    assert _WORKER_FS is not None and _WORKER_BUDGET is not None
    budget = _WORKER_BUDGET
    t1 = time.time()
    status, err = "ok", ""
    try:
        res = run_algorithm(job["algorithm"], _WORKER_FS, job["target"], budget, job["seed"])
    except Exception as e:  # noqa: BLE001 — record and continue
        status, err = "error", str(e)
        res = _empty_res(budget)
    runtime = time.time() - t1
    meta = dict(job["meta"])
    meta.update(
        {
            "FHR": res.get("fhr", 0.0),
            "BestFit": res.get("best_fitness", 0.0),
            "FSC": res.get("fsc", 0),
            "Diversity": res.get("diversity", 0.0),
            "Contrib": res.get("contrib", 0.0),
            "CV": 1.0 - float(res.get("best_fitness", 0.0)),
            "FFG": res.get("ffg", 10**9),
            "FFG_evals": res.get("ffg_evals", res.get("n_evals", 0)),
            "n_evals": res.get("n_evals", 0),
            "n_foods": res.get("avg_n_foods", 0.0),
            "runtime": runtime,
            "status": status,
            "error": err,
        }
    )
    return meta


def run_job_batch(
    jobs: Sequence[dict],
    fs,
    budget: RunBudget,
    workers: int,
    rows: List[dict],
    done_keys: set,
    key_from_row: Callable[[dict], tuple],
    planned: int,
    partial_path: str,
    log_dir: str,
    t0: float,
    checkpoint_every: int = 50,
    print_every: int = 10,
    log_prefix: str = "",
) -> int:
    """
    Run jobs serially (workers=1) or via ProcessPoolExecutor.
    Appends to `rows`, updates `done_keys`, checkpoints to partial_path.
    Returns number of newly completed jobs this call.
    """
    if not jobs:
        return 0

    sess_done = 0
    done = len(done_keys)

    def _consume(row: dict) -> None:
        nonlocal sess_done, done
        rows.append(row)
        done_keys.add(key_from_row(row))
        done += 1
        sess_done += 1
        el = time.time() - t0
        eta_h = (el / sess_done) * max(planned - done, 0) / 3600.0
        if sess_done <= 3 or done % print_every == 0 or done == planned:
            print(
                f"  [{done}/{planned}] {log_prefix}{row.get('algorithm', '?'):12s} "
                f"{row.get('target_id', '?')} r{row.get('rep', '?')} "
                f"FHR={float(row.get('FHR', 0)):.0f} Fit={float(row.get('BestFit', 0)):.3f} "
                f"evals={row.get('n_evals', 0)} t={float(row.get('runtime', 0)):.1f}s "
                f"{row.get('status', '')} ETA={eta_h:.1f}h",
                flush=True,
            )
        if done % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            append_log(log_dir, f"checkpoint {done}/{planned} -> {partial_path}")

    if workers <= 1:
        _init_worker(fs, budget)
        for job in jobs:
            _consume(algo_worker_job(job))
        return sess_done

    print(f"  parallel workers={workers} jobs={len(jobs)}", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(fs, budget),
    ) as ex:
        futs = [ex.submit(algo_worker_job, job) for job in jobs]
        for fut in as_completed(futs):
            _consume(fut.result())
    return sess_done
