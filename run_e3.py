#!/usr/bin/env python3
"""
E3 — Constraint dimensionality experiment (production / fair budget).

For each dim in {6,9,12,16}: project to the first `dim` NHANES nutrients
(canonical column order). Representative targets that omit some nutrients
are completed with [0, empirical_meal_cap] so high-dim cells are well-defined.
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
from benchmark_core import RunBudget, write_env
from e1_scaling import Target, load_nhanes_full, sample_food_space
from feasibility_geometry import estimate_feasible_density
from parallel_jobs import default_workers, run_job_batch
from run_benchmark import load_representative_targets
from sparsity_ops import empirical_meal_caps


def expand_target_to_full(
    tgt: Target,
    full_names: list[str],
    caps: np.ndarray,
) -> Target:
    """Map a partial target onto the full nutrient list; fill missing with [0, cap]."""
    name_to_i = {n: i for i, n in enumerate(tgt.nutrient_names)}
    lowers, uppers = [], []
    for j, n in enumerate(full_names):
        if n in name_to_i:
            i = name_to_i[n]
            lowers.append(float(tgt.lower_bounds[i]))
            uppers.append(float(tgt.upper_bounds[i]))
        else:
            lowers.append(0.0)
            uppers.append(float(caps[j]) if np.isfinite(caps[j]) and caps[j] > 0 else 1e6)
    arr = np.column_stack([lowers, uppers]).astype(np.float32)
    return Target(
        target_id=tgt.target_id,
        lower_bounds=arr[:, 0],
        upper_bounds=arr[:, 1],
        nutrient_names=list(full_names),
        col_idx=np.arange(len(full_names), dtype=np.int32),
        target_array=arr,
    )


def project_dim(tgt: Target, dim: int) -> Target:
    d = min(dim, tgt.n_dims)
    arr = tgt.target_array[:d].copy()
    return Target(
        target_id=tgt.target_id,
        lower_bounds=arr[:, 0],
        upper_bounds=arr[:, 1],
        nutrient_names=tgt.nutrient_names[:d],
        col_idx=np.arange(d, dtype=np.int32),
        target_array=arr,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="main")
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
        pop_size=int(cfg.get("population_size", 500)),
        max_evaluations=int(cfg.get("max_evaluations", 100000)),
        n_generations=int(cfg.get("max_generations", 10000)),
        early_stop_feasible=cfg.get("early_stop_feasible", None),
    )
    algorithms = cfg["algorithms"]
    dimensions = [int(d) for d in cfg.get("dimensions", [6, 9, 12, 16])]
    n_targets = int(cfg.get("targets_total", 12))
    n_repeats = int(cfg.get("repeats", 10))
    seed = int(cfg.get("seed", 42))
    density_samples = int(cfg.get("density_samples", 0))
    exp_name = cfg.get("experiment", "e3_dimension")
    result_class = cfg.get("result_class", "PRIMARY")
    workers = default_workers(args.workers, cfg)

    sc = cfg["scales"][0]
    size = int(sc["size"])
    out_path = os.path.join(out_dir, cfg.get("output_name", "e3_dimension_full.csv"))
    partial_path = out_path + ".partial.csv"

    print("=" * 70)
    print(f"E3 Dimension | stage={args.stage} | class={result_class}")
    print(f"budget pop={budget.pop_size} evals={budget.max_evaluations} workers={workers}")
    print(f"dims={dimensions} targets={n_targets} reps={n_repeats} algos={len(algorithms)} N={size}")
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
    aligned = []
    for t in raw_targets:
        common_names = [n for n in t.nutrient_names if n in nutrients]
        if len(common_names) < 4:
            continue
        jpos = [t.nutrient_names.index(n) for n in common_names]
        idxs = [nutrients.index(n) for n in common_names]
        arr = t.target_array[jpos]
        aligned.append(Target(
            t.target_id, arr[:, 0], arr[:, 1], common_names,
            np.array(idxs, np.int32), arr.astype(np.float32),
        ))
    aligned = aligned[:n_targets]

    all_col = np.arange(len(nutrients), dtype=np.int32)
    caps = empirical_meal_caps(
        fs.nutrients, all_col, fs.food_ids, fs.meal_tags,
        fs.min_foods, fs.max_foods, fs.has_meal_constraint,
        n_samples=int(cfg.get("cap_samples", 600)),
        percentile=float(cfg.get("cap_percentile", 99.0)),
        seed=seed,
    )
    base_full = [expand_target_to_full(t, nutrients, caps) for t in aligned]
    print(f"base targets={len(base_full)} expanded to {len(nutrients)} nutrients")

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
                done_keys.add((int(r["dim"]), r["algorithm"], r["target_id"], int(r["rep"])))
            print(f"Resume: loaded {len(rows)} rows from {best}")

    total = len(dimensions) * len(base_full) * len(algorithms) * n_repeats
    t0 = time.time()
    print(f"planned_runs={total} (resume_skip={len(done_keys)})")

    def key_from_row(r):
        return (int(r["dim"]), r["algorithm"], r["target_id"], int(r["rep"]))

    for dim in dimensions:
        print(f"\n--- dim={dim} ---")
        fs_dim = type(fs)(
            nutrients=fs.nutrients[:, :dim].astype(np.float32),
            meal_tags=fs.meal_tags,
            categories=fs.categories,
            food_ids=fs.food_ids,
            min_foods=fs.min_foods,
            max_foods=fs.max_foods,
            has_meal_constraint=fs.has_meal_constraint,
        )
        density_by_tid = {}
        jobs = []
        for tgt_full in base_full:
            tgt = project_dim(tgt_full, dim)
            geom = None
            if density_samples > 0:
                geom = estimate_feasible_density(
                    fs_dim.nutrients, fs_dim.meal_tags, fs_dim.food_ids,
                    tgt.target_array, list(range(tgt.n_dims)),
                    min_foods=fs_dim.min_foods, max_foods=fs_dim.max_foods,
                    has_meal_constraint=fs_dim.has_meal_constraint,
                    n_samples=density_samples,
                    seed=seed + dim * 1000 + hash(tgt.target_id) % 10000,
                )
                density_by_tid[tgt.target_id] = geom
                print(f"  {tgt.target_id}: ρ={geom.density:.5f} meanV={geom.mean_violation:.1f}")

            for alg in algorithms:
                for rep in range(n_repeats):
                    key = (dim, alg, tgt.target_id, int(rep))
                    if key in done_keys:
                        continue
                    run_seed = make_seed_local(seed, exp_name, dim, fs.n_foods, tgt.target_id, alg, rep)
                    geom = density_by_tid.get(tgt.target_id)
                    jobs.append({
                        "algorithm": alg,
                        "seed": run_seed,
                        "target": tgt,
                        "meta": {
                            "experiment": exp_name,
                            "dataset": sc.get("dataset", "nhanes_sample"),
                            "scale": fs.n_foods,
                            "dim": dim,
                            "sparsity": float(cfg.get("sparsity", 1.0)),
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
                        },
                    })

        run_job_batch(
            jobs, fs_dim, budget, workers, rows, done_keys, key_from_row,
            total, partial_path, log_dir, t0,
            checkpoint_every=50, print_every=10,
            log_prefix=f"d={dim} ",
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
        summ = df.groupby(["dim", "algorithm"])[["FHR", "BestFit", "FFG_evals", "runtime"]].mean()
        summ.to_csv("results/tables/e3_summary_by_dim_algo.csv")
        print(summ.round(3))

    print(f"\nSaved {len(df)} rows -> {out_path} in {time.time()-t0:.1f}s")


def make_seed_local(seed, exp_name, dim, n_foods, target_id, alg, rep):
    from benchmark_core import make_seed
    return make_seed(seed, f"{exp_name}_d{dim}", n_foods, target_id, alg, rep)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
