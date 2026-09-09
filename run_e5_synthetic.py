#!/usr/bin/env python3
"""
E5: Synthetic RCFA external-validity experiment (compact).

Preserves interval-box binary subset selection without food semantics.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yaml

from benchmark_core import RunBudget, make_seed, write_env
from e1_scaling import Target
from parallel_jobs import default_workers, run_job_batch


@dataclass
class SynthFoodSpace:
    nutrients: np.ndarray
    meal_tags: np.ndarray
    food_ids: np.ndarray
    min_foods: int = 5
    max_foods: int = 15
    has_meal_constraint: bool = False
    n_foods: int = 0

    def __post_init__(self):
        self.n_foods = int(self.nutrients.shape[0])
        if self.food_ids is None or len(self.food_ids) != self.n_foods:
            self.food_ids = np.arange(self.n_foods, dtype=np.int32)


def make_synthetic_matrix(n: int, m: int, structure: str, rng: np.random.Generator) -> np.ndarray:
    if structure == "dense":
        A = rng.lognormal(mean=0.0, sigma=0.8, size=(n, m))
    elif structure == "sparse":
        A = rng.lognormal(mean=0.0, sigma=0.8, size=(n, m))
        mask = rng.random((n, m)) < 0.85
        A[mask] = 0.0
        # ensure each column has support
        for j in range(m):
            if A[:, j].sum() <= 0:
                A[rng.integers(0, n), j] = float(rng.lognormal(0.0, 0.8))
    elif structure == "correlated":
        base = rng.lognormal(mean=0.0, sigma=0.7, size=(n, 1))
        noise = rng.lognormal(mean=0.0, sigma=0.4, size=(n, m))
        A = base * (0.6 + 0.4 * rng.random(m)) + 0.3 * noise
    else:
        raise ValueError(structure)
    return A.astype(np.float32)


def make_interval_box(A: np.ndarray, width_factor: float, rng: np.random.Generator):
    """Build box around a random feasible-ish reference meal sum, then scale width."""
    n, m = A.shape
    k = int(rng.integers(5, 16))
    idx = rng.choice(n, size=k, replace=False)
    center = A[idx].sum(axis=0)
    # base half-width relative to center
    half = np.maximum(0.15 * np.abs(center), 1.0)
    half = half * float(width_factor)
    lo = np.maximum(center - half, 0.0)
    hi = center + half
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e5_synthetic.yaml")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg.get("output_dir", "results/raw")
    log_dir = cfg.get("log_dir", "results/logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    write_env("results/reproducibility/env.json")

    Ns = [int(x) for x in cfg.get("N_list", [5000, 10000, 25000])]
    dims = [int(x) for x in cfg.get("dims", [6, 12, 16])]
    factors = [float(x) for x in cfg.get("width_factors", [0.5, 1.0, 1.5])]
    structures = cfg.get("structures", ["dense", "sparse", "correlated"])
    # Compact default: one structure if specified
    if cfg.get("structure_primary"):
        structures = [cfg["structure_primary"]]
    algorithms = cfg["algorithms"]
    n_inst = int(cfg.get("instances_per_cell", 2))
    n_repeats = int(cfg.get("repeats", 5))
    seed0 = int(cfg.get("seed", 42))
    workers = default_workers(args.workers, cfg)
    out_path = os.path.join(out_dir, cfg.get("output_name", "e5_synthetic.csv"))
    partial_path = out_path + ".partial.csv"

    budget = RunBudget(
        pop_size=int(cfg.get("population_size", 500)),
        max_evaluations=int(cfg.get("max_evaluations", 100000)),
        n_generations=int(cfg.get("max_generations", 10000)),
        early_stop_feasible=None,
    )

    rows, done_keys = [], set()

    def key_from_row(r):
        return (r["structure"], int(r["scale"]), int(r["dim"]), float(r["sparsity"]),
                r["target_id"], r["algorithm"], int(r["rep"]))

    for p in (partial_path, out_path):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            prev = pd.read_csv(p)
            if len(prev):
                rows = prev.to_dict("records")
                for r in rows:
                    done_keys.add(key_from_row(r))
                print(f"Resume {len(rows)} from {p}")
            break

    # Build instance catalog
    instances = []
    for structure in structures:
        for N in Ns:
            for m in dims:
                for f in factors:
                    for inst in range(n_inst):
                        rng = np.random.default_rng(seed0 + hash((structure, N, m, f, inst)) % (2**31))
                        A = make_synthetic_matrix(N, m, structure, rng)
                        lo, hi = make_interval_box(A, f, rng)
                        tid = f"syn_{structure}_N{N}_m{m}_f{f}_i{inst}"
                        ta = np.stack([lo, hi], axis=1)
                        names = [f"c{j}" for j in range(m)]
                        tgt = Target(
                            target_id=tid,
                            lower_bounds=lo,
                            upper_bounds=hi,
                            nutrient_names=names,
                            col_idx=np.arange(m, dtype=np.int32),
                            target_array=ta,
                        )
                        fs = SynthFoodSpace(
                            nutrients=A,
                            meal_tags=np.zeros(N, dtype=np.int8),
                            food_ids=np.arange(N, dtype=np.int32),
                            min_foods=5,
                            max_foods=15,
                            has_meal_constraint=False,
                        )
                        instances.append((structure, N, m, f, tgt, fs))

    planned = len(instances) * len(algorithms) * n_repeats
    print(f"E5 instances={len(instances)} planned_runs={planned} skip={len(done_keys)}")
    t0 = time.time()

    # One batch per synthetic instance (each has its own coefficient matrix A)
    for structure, N, m, f, tgt, fs in instances:
        jobs = []
        for alg in algorithms:
            for rep in range(n_repeats):
                key = (structure, N, m, f, tgt.target_id, alg, rep)
                if key in done_keys:
                    continue
                run_seed = make_seed(seed0, f"e5_{structure}", N, tgt.target_id, alg, rep)
                jobs.append({
                    "algorithm": alg,
                    "seed": run_seed,
                    "target": tgt,
                    "meta": {
                        "experiment": "e5_synthetic",
                        "dataset": "synthetic",
                        "structure": structure,
                        "scale": N,
                        "dim": m,
                        "sparsity": f,
                        "algorithm": alg,
                        "target_id": tgt.target_id,
                        "rep": rep,
                        "seed": run_seed,
                        "pop_size": budget.pop_size,
                        "max_evaluations": budget.max_evaluations,
                        "result_class": cfg.get("result_class", "SYNTHETIC"),
                    },
                })
        print(f"\n--- {tgt.target_id} jobs={len(jobs)} ---")
        run_job_batch(
            jobs, fs, budget, workers, rows, done_keys, key_from_row,
            planned, partial_path, log_dir, t0,
            checkpoint_every=40, print_every=15,
            log_prefix=f"{tgt.target_id} ",
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    if os.path.exists(partial_path):
        try:
            os.remove(partial_path)
        except OSError:
            pass
    if len(df):
        summ = df.groupby(["structure", "sparsity", "algorithm"])["FHR"].mean().unstack()
        print(summ.round(3))
        os.makedirs("results/tables", exist_ok=True)
        summ.to_csv("results/tables/e5_fhr_by_sparsity_algo.csv")
    print(f"Saved {len(df)} -> {out_path} time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
