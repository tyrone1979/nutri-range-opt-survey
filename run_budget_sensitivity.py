#!/usr/bin/env python3
"""Budget-sensitivity: NHANES-10K, E2 f=0.5, vary max_evaluations only."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

import common as cm
from benchmark_core import RunBudget, make_seed, write_env
from e1_scaling import Target, load_nhanes_full, sample_food_space
from parallel_jobs import default_workers, run_job_batch
from run_benchmark import align_target_to_space, load_representative_targets
from sparsity_ops import empirical_meal_caps, prepare_sparsity_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e_budget.yaml")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = cfg.get("output_dir", "results/raw")
    log_dir = cfg.get("log_dir", "results/logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    write_env("results/reproducibility/env.json")

    algorithms = cfg["algorithms"]
    budgets_list = [int(x) for x in cfg["evaluation_budgets"]]
    n_targets = int(cfg.get("targets_total", 12))
    n_repeats = int(cfg.get("repeats", 10))
    seed = int(cfg.get("seed", 42))
    fac = float(cfg.get("sparsity_factor", 0.5))
    size = int(cfg["scales"][0]["size"])
    workers = default_workers(args.workers, cfg)
    out_path = os.path.join(out_dir, cfg.get("output_name", "e_budget_f05.csv"))
    partial_path = out_path + ".partial.csv"
    exp_name = cfg.get("experiment", "e_budget")
    result_class = cfg.get("result_class", "BUDGET_SENSITIVITY")

    matrix, meal_tags, nutrients, _ = cm.load_data(
        "data/nhanes.csv", "data/food_type.csv", "data/target.csv"
    )
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    fs = sample_food_space(full, size, seed=seed)

    raw_targets = load_representative_targets(cfg.get("targets_file"), n_targets)
    targets = []
    for t in raw_targets:
        at = align_target_to_space(t, nutrients)
        if at is not None and at.n_dims >= 4:
            targets.append(at)
    targets = targets[:n_targets]

    rows: list = []
    done_keys: set = set()

    def key_from_row(r):
        return (int(r["max_evaluations"]), r["algorithm"], r["target_id"], int(r["rep"]))

    for p in (partial_path, out_path):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            prev = pd.read_csv(p)
            if len(prev):
                rows = prev.to_dict("records")
                for r in rows:
                    done_keys.add(key_from_row(r))
                print(f"Resume: {len(rows)} rows from {p}")
            break

    planned = len(budgets_list) * len(targets) * len(algorithms) * n_repeats
    t0 = time.time()
    print(f"Budget sensitivity planned={planned} skip={len(done_keys)} workers={workers}")

    # Precompute scaled targets once
    scaled_targets = []
    for tgt in targets:
        caps = empirical_meal_caps(
            fs.nutrients, tgt.col_idx, fs.food_ids, fs.meal_tags,
            fs.min_foods, fs.max_foods, True,
            n_samples=int(cfg.get("cap_samples", 600)),
            percentile=float(cfg.get("cap_percentile", 99.0)),
            seed=seed + tgt.n_dims,
        )
        scaled_arr, smeta = prepare_sparsity_target(tgt.target_array, fac, caps)
        adj = Target(
            target_id=tgt.target_id,
            lower_bounds=scaled_arr[:, 0],
            upper_bounds=scaled_arr[:, 1],
            nutrient_names=tgt.nutrient_names,
            col_idx=tgt.col_idx,
            target_array=scaled_arr,
        )
        scaled_targets.append((adj, smeta))

    for max_evals in budgets_list:
        budget = RunBudget(
            pop_size=int(cfg.get("population_size", 500)),
            max_evaluations=max_evals,
            n_generations=int(cfg.get("max_generations", 10000)),
            early_stop_feasible=None,
        )
        jobs = []
        for adj, smeta in scaled_targets:
            for alg in algorithms:
                for rep in range(n_repeats):
                    key = (max_evals, alg, adj.target_id, rep)
                    if key in done_keys:
                        continue
                    run_seed = make_seed(
                        seed, f"{exp_name}_{max_evals}", fs.n_foods, adj.target_id, alg, rep
                    )
                    jobs.append({
                        "algorithm": alg,
                        "seed": run_seed,
                        "target": adj,
                        "meta": {
                            "experiment": exp_name,
                            "dataset": "nhanes_sample",
                            "scale": fs.n_foods,
                            "dim": adj.n_dims,
                            "sparsity": fac,
                            "algorithm": alg,
                            "target_id": adj.target_id,
                            "rep": rep,
                            "seed": run_seed,
                            "pop_size": budget.pop_size,
                            "max_evaluations": max_evals,
                            "result_class": result_class,
                            "n_unbounded_capped": smeta["n_unbounded_capped"],
                        },
                    })
        print(f"\n--- max_evaluations={max_evals} jobs={len(jobs)} ---")
        run_job_batch(
            jobs, fs, budget, workers, rows, done_keys, key_from_row,
            planned, partial_path, log_dir, t0,
            checkpoint_every=50, print_every=20,
            log_prefix=f"B={max_evals} ",
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    if os.path.exists(partial_path):
        try:
            os.remove(partial_path)
        except OSError:
            pass
    os.makedirs("results/tables", exist_ok=True)
    if len(df):
        summ = df.groupby(["max_evaluations", "algorithm"])[["FHR", "FFG_evals", "n_evals"]].mean()
        summ.to_csv("results/tables/e_budget_summary.csv")
        print(summ.round(3))
    print(f"Saved {len(df)} -> {out_path}  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
