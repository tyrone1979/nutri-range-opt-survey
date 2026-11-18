#!/usr/bin/env python3
"""
E2 — Feasible-region sparsity experiment (production / fair budget).

Uses run_benchmark.run_algorithm + sanitized sparsity scaling.
Supports --workers process pool + checkpoint/resume.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import yaml

import common as cm
from benchmark_core import RunBudget, make_seed, write_env
from e1_scaling import Target, load_nhanes_full, sample_food_space
from feasibility_geometry import estimate_feasible_density
from parallel_jobs import default_workers, run_job_batch
from run_benchmark import align_target_to_space, load_representative_targets
from sparsity_ops import empirical_meal_caps, prepare_sparsity_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "results/raw")
    log_dir = cfg.get("log_dir", "results/logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    write_env("results/reproducibility/env.json")

    budget = RunBudget(
        pop_size=int(cfg.get("population_size", 200)),
        max_evaluations=int(cfg.get("max_evaluations", 20000)),
        n_generations=int(cfg.get("max_generations", 10000)),
        early_stop_feasible=cfg.get("early_stop_feasible", None),
    )
    algorithms = cfg["algorithms"]
    factors = [float(x) for x in cfg.get("sparsity_factors", [0.5, 1.0, 1.5])]
    n_targets = int(cfg.get("targets_total", 12))
    n_repeats = int(cfg.get("repeats", 5))
    seed = int(cfg.get("seed", 42))
    density_samples = int(cfg.get("density_samples", 1000))
    exp_name = cfg.get("experiment", "e2_sparsity")
    result_class = cfg.get("result_class", "PILOT")
    workers = default_workers(args.workers, cfg)

    sc = cfg["scales"][0]
    size = int(sc["size"])
    out_path = os.path.join(out_dir, cfg.get("output_name", "e2_sparsity.csv"))
    partial_path = out_path + ".partial.csv"

    print("=" * 70)
    print(f"E2 Sparsity | stage={args.stage} | class={result_class}")
    print(f"budget pop={budget.pop_size} evals={budget.max_evaluations} workers={workers}")
    print(f"factors={factors} targets={n_targets} reps={n_repeats} algos={len(algorithms)} N={size}")
    print("=" * 70)

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

    done_keys = set()
    rows = []
    resume_candidates = [p for p in (partial_path, out_path)
                         if os.path.exists(p) and os.path.getsize(p) > 0]
    if resume_candidates:
        best = max(resume_candidates, key=lambda p: len(pd.read_csv(p)))
        prev = pd.read_csv(best)
        if len(prev):
            rows = prev.to_dict("records")
            for r in rows:
                done_keys.add((float(r["sparsity"]), r["algorithm"], r["target_id"], int(r["rep"])))
            print(f"Resume: loaded {len(rows)} rows from {best}")

    total = len(factors) * len(targets) * len(algorithms) * n_repeats
    t0 = time.time()
    print(f"planned_runs={total} (resume_skip={len(done_keys)})")

    def key_from_row(r):
        return (float(r["sparsity"]), r["algorithm"], r["target_id"], int(r["rep"]))

    for fac in factors:
        print(f"\n--- sparsity_factor={fac} ---")
        jobs = []
        for tgt in targets:
            caps = empirical_meal_caps(
                fs.nutrients, tgt.col_idx, fs.food_ids, fs.meal_tags,
                fs.min_foods, fs.max_foods, fs.has_meal_constraint,
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
            geom = None
            if density_samples > 0:
                mat = fs.nutrients[:, tgt.col_idx]
                geom = estimate_feasible_density(
                    mat, fs.meal_tags, fs.food_ids, scaled_arr, list(range(adj.n_dims)),
                    min_foods=fs.min_foods, max_foods=fs.max_foods,
                    has_meal_constraint=fs.has_meal_constraint,
                    n_samples=density_samples,
                    seed=make_seed(
                        seed,
                        f"{exp_name}_density_f{fac}",
                        fs.n_foods,
                        tgt.target_id,
                        "density",
                        0,
                    ),
                )
                print(f"  {tgt.target_id}: capped={smeta['n_unbounded_capped']} "
                      f"ρ={geom.density:.5f} meanV={geom.mean_violation:.1f}")

            for alg in algorithms:
                for rep in range(n_repeats):
                    key = (float(fac), alg, tgt.target_id, int(rep))
                    if key in done_keys:
                        continue
                    run_seed = make_seed(seed, f"{exp_name}_f{fac}", fs.n_foods, tgt.target_id, alg, rep)
                    jobs.append({
                        "algorithm": alg,
                        "seed": run_seed,
                        "target": adj,
                        "meta": {
                            "experiment": exp_name,
                            "dataset": sc.get("dataset", "nhanes_sample"),
                            "scale": fs.n_foods,
                            "dim": adj.n_dims,
                            "sparsity": fac,
                            "algorithm": alg,
                            "target_id": tgt.target_id,
                            "rep": rep,
                            "seed": run_seed,
                            "pop_size": budget.pop_size,
                            "max_evaluations": budget.max_evaluations,
                            "result_class": result_class,
                            "feasible_density": geom.density if geom else np.nan,
                            "density_mean_violation": geom.mean_violation if geom else np.nan,
                            "density_median_violation": geom.median_violation if geom else np.nan,
                            "density_n_samples": geom.n_samples if geom else 0,
                            "n_unbounded_capped": smeta["n_unbounded_capped"],
                        },
                    })

        run_job_batch(
            jobs, fs, budget, workers, rows, done_keys, key_from_row,
            total, partial_path, log_dir, t0,
            checkpoint_every=50, print_every=20,
            log_prefix=f"f={fac} ",
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
        summ = df.groupby(["sparsity", "algorithm"])[["FHR", "BestFit", "FFG_evals", "runtime"]].mean()
        summ.to_csv("results/tables/e2_summary_by_sparsity_algo.csv")
        dens = df.drop_duplicates(["sparsity", "target_id"])[
            ["sparsity", "target_id", "feasible_density", "density_mean_violation", "n_unbounded_capped"]
        ]
        dens.to_csv("results/tables/e2_density_by_target.csv", index=False)
        print(summ.round(3))

    print(f"\nSaved {len(df)} rows -> {out_path}")
    print(f"Total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
