#!/usr/bin/env python3
"""Production NutriOpt-Bench runner with fair budgets and logging."""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import yaml

from benchmark_core import RunBudget, append_log, collect_software_versions, empty_metrics, make_seed, set_global_seed, write_env
from common import compute_max_categories
from e1_scaling import (
    ExperimentResult,
    FoodSpace,
    Target,
    get_food_space,
    load_allrecipes,
    load_foods_com,
    load_nhanes_full,
    generate_small_dataset_targets,
    sample_food_space,
)
from feasibility_geometry import estimate_feasible_density


def load_representative_targets(path: str, n: int, nutrient_names_full=None) -> list:
    """Load targets from representative CSV (target_json column)."""
    import json
    df = pd.read_csv(path)
    # Support multiple schemas
    json_col = "target_json" if "target_json" in df.columns else None
    targets = []
    for i, row in df.head(n).iterrows():
        if json_col:
            td = json.loads(row[json_col]) if isinstance(row[json_col], str) else row[json_col]
        else:
            td = eval(row["target"])
        nutrients = list(td.keys())
        arr = np.array([[td[k][0], td[k][1]] for k in nutrients], dtype=np.float32)
        tid = f"rep_{int(row.get('representative_id', row.get('rep_id', i))):03d}"
        targets.append(Target(
            target_id=tid,
            lower_bounds=arr[:, 0],
            upper_bounds=arr[:, 1],
            nutrient_names=nutrients,
            col_idx=np.arange(len(nutrients), dtype=np.int32),  # remapped later
            target_array=arr,
        ))
    return targets


def align_target_to_space(target: Target, space_nutrient_names: list) -> Target | None:
    """Map target nutrients onto FoodSpace nutrient column order."""
    common = [n for n in target.nutrient_names if n in space_nutrient_names]
    if not common:
        # try synonym map for recipe datasets
        syn = {"calorie": "calories", "carb": "carbs", "calories": "calorie", "carbs": "carb"}
        mapped = []
        for n in target.nutrient_names:
            m = syn.get(n, n)
            if m in space_nutrient_names:
                mapped.append((n, m))
        if not mapped:
            return None
        lowers, uppers, names, idxs = [], [], [], []
        for orig, m in mapped:
            j = target.nutrient_names.index(orig)
            lowers.append(target.lower_bounds[j])
            uppers.append(target.upper_bounds[j])
            names.append(m)
            idxs.append(space_nutrient_names.index(m))
        arr = np.column_stack([lowers, uppers]).astype(np.float32)
        return Target(target.target_id, np.array(lowers, np.float32), np.array(uppers, np.float32),
                      names, np.array(idxs, np.int32), arr)

    idxs = [space_nutrient_names.index(n) for n in common]
    jpos = [target.nutrient_names.index(n) for n in common]
    arr = target.target_array[jpos]
    return Target(target.target_id, arr[:, 0], arr[:, 1], common, np.array(idxs, np.int32), arr)


def run_algorithm(name, fs: FoodSpace, target: Target, budget: RunBudget, seed: int) -> dict:
    set_global_seed(seed)
    matrix = fs.nutrients
    meal_tags = fs.meal_tags
    all_valid = fs.food_ids
    max_cat = compute_max_categories(all_valid)
    ta = target.target_array
    col_idx = list(range(len(target.nutrient_names)))
    # Remap matrix columns: build view with only target nutrients in order
    mat = matrix[:, target.col_idx]

    kwargs = dict(
        pop_size=budget.pop_size,
        n_generations=budget.n_generations,
        max_evals=budget.max_evaluations,
        min_foods=fs.min_foods,
        max_foods=fs.max_foods,
        has_meal_constraint=fs.has_meal_constraint,
        early_stop_feasible=budget.early_stop_feasible,
    )

    if name in ("BMOEA", "ACMO-Diet", "MA"):
        from baseline_algorithms import BMOEA, ACMODiet, MemeticAlgorithm
        cls = {"BMOEA": BMOEA, "ACMO-Diet": ACMODiet, "MA": MemeticAlgorithm}[name]
        return cls(mat, meal_tags, all_valid, max_cat).run(ta, col_idx, **kwargs)[0]

    if name in ("NSGA2", "RVEA", "CTAEA", "SPEA2", "NSGA3", "MOEAD"):
        from baseline_algorithms import run_pymoo_algorithm
        return run_pymoo_algorithm(
            name, mat, ta, col_idx, meal_tags, all_valid, max_cat,
            pop_size=budget.pop_size, n_generations=budget.n_generations,
            min_foods=fs.min_foods, max_foods=fs.max_foods,
            has_meal_constraint=fs.has_meal_constraint,
            seed=seed, max_evals=budget.max_evaluations,
            early_stop_feasible=budget.early_stop_feasible,
        )

    if name in ("Random", "Greedy", "RandomRepair"):
        from simple_baselines import RandomSearch, GreedyConstruct, RandomRepair
        cls = {"Random": RandomSearch, "Greedy": GreedyConstruct, "RandomRepair": RandomRepair}[name]
        return cls().run(
            mat, meal_tags, all_valid, ta, col_idx,
            pop_size=budget.pop_size, max_evaluations=budget.max_evaluations,
            n_generations=budget.n_generations, min_foods=fs.min_foods,
            max_foods=fs.max_foods, has_meal_constraint=fs.has_meal_constraint,
            early_stop_feasible=budget.early_stop_feasible, seed=seed,
        )[0]

    raise ValueError(f"Unknown algorithm {name}")


def nhanes_nutrient_names():
    # After common.load_data drops high-missing cols; vitamin_d kept (<0.5 missing)
    return [
        "calorie", "protein", "carb", "sugar", "fiber", "saturated_fat", "cholesterol",
        "folic_acid", "vitamin_b12", "vitamin_c", "vitamin_d", "calcium", "phosphorus",
        "potassium", "iron", "sodium",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="pilot", choices=["profile", "pilot", "main", "custom"])
    parser.add_argument("--workers", type=int, default=None,
                        help="Process-pool size (default: min(32, ncpu-2) or NUTRIOPT_WORKERS)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from parallel_jobs import default_workers, run_job_batch

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
    n_targets = int(cfg.get("targets_total", 3))
    n_repeats = int(cfg.get("repeats", 1))
    seed = int(cfg.get("seed", 42))
    exp_name = cfg.get("experiment", "e1_scaling")
    workers = default_workers(args.workers, cfg)

    print("=" * 70)
    print(f"NutriOpt-Bench production runner | stage={args.stage} | exp={exp_name}")
    print(f"budget: pop={budget.pop_size} max_evals={budget.max_evaluations} early_stop={budget.early_stop_feasible}")
    print(f"algorithms={algorithms} targets={n_targets} repeats={n_repeats} workers={workers}")
    print("=" * 70)

    full = load_nhanes_full()
    import common as cm
    matrix, meal_tags, nutrients, _ = cm.load_data("data/nhanes.csv", "data/food_type.csv", "data/target.csv")
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    nh_names = nutrients

    rep_path = cfg.get("targets_file", "data/representative_targets.csv")
    raw_targets = load_representative_targets(rep_path, n_targets)

    out_path = os.path.join(out_dir, cfg.get("output_name", f"{exp_name}.csv"))
    partial_path = out_path + ".partial.csv"
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
                done_keys.add((r["dataset"], int(r["scale"]), r["algorithm"],
                               r["target_id"], int(r["rep"])))
            print(f"Resume: loaded {len(rows)} rows from {best}")

    t0 = time.time()
    scales = cfg.get("scales", [{"dataset": "nhanes_sample", "size": 1500, "label": "NHANES-1.5K"}])

    planned = 0
    plan_cache = []
    for sc in scales:
        ds = sc["dataset"]
        if ds in ("nhanes_sample", "nhanes_full"):
            n_tgt = n_targets
        elif ds in ("allrecipes", "foods_com"):
            n_tgt = min(n_targets, 6)
        else:
            raise ValueError(ds)
        planned += n_tgt * len(algorithms) * n_repeats
        plan_cache.append((sc, ds))
    print(f"planned_runs≈{planned} (resume_skip={len(done_keys)})")

    def key_from_row(r):
        return (r["dataset"], int(r["scale"]), r["algorithm"], r["target_id"], int(r["rep"]))

    for sc, ds in plan_cache:
        if ds == "nhanes_sample":
            fs = sample_food_space(full, int(sc["size"]), seed=seed)
            space_names = nh_names
        elif ds == "nhanes_full":
            fs = full
            space_names = nh_names
        elif ds == "allrecipes":
            fs = load_allrecipes()
            space_names = ["calories", "protein", "carbs", "fat"]
        elif ds == "foods_com":
            fs = load_foods_com()
            space_names = ["calories", "protein", "carbs", "fat"]
        else:
            raise ValueError(ds)

        if ds in ("allrecipes", "foods_com"):
            targets = generate_small_dataset_targets(fs, n_targets=min(n_targets, 6), seed=seed)
        else:
            targets = []
            for t in raw_targets:
                at = align_target_to_space(t, space_names)
                if at is not None and at.n_dims >= 4:
                    targets.append(at)
            targets = targets[:n_targets]

        print(f"\nDataset {sc.get('label', ds)} n={fs.n_foods} targets={len(targets)}")

        density_samples = int(cfg.get("density_samples", 0))
        density_by_tid = {}
        if density_samples > 0:
            for tgt in targets:
                mat_view = fs.nutrients[:, tgt.col_idx]
                geom = estimate_feasible_density(
                    mat_view, fs.meal_tags, fs.food_ids, tgt.target_array,
                    list(range(tgt.n_dims)),
                    min_foods=fs.min_foods, max_foods=fs.max_foods,
                    has_meal_constraint=fs.has_meal_constraint,
                    n_samples=density_samples,
                    seed=seed + fs.n_foods + hash(tgt.target_id) % 10000,
                )
                density_by_tid[tgt.target_id] = geom
                print(f"  density[{tgt.target_id}]: ρ={geom.density:.4f} "
                      f"meanV={geom.mean_violation:.2f} (n={geom.n_samples})")

        jobs = []
        for tgt in targets:
            geom = density_by_tid.get(tgt.target_id)
            for alg in algorithms:
                for rep in range(n_repeats):
                    key = (ds, int(fs.n_foods), alg, tgt.target_id, int(rep))
                    if key in done_keys:
                        continue
                    run_seed = make_seed(seed, exp_name, fs.n_foods, tgt.target_id, alg, rep)
                    jobs.append({
                        "algorithm": alg,
                        "seed": run_seed,
                        "target": tgt,
                        "meta": {
                            "experiment": exp_name,
                            "dataset": ds,
                            "scale": fs.n_foods,
                            "dim": tgt.n_dims,
                            "sparsity": float(cfg.get("sparsity", 1.0)),
                            "algorithm": alg,
                            "target_id": tgt.target_id,
                            "rep": rep,
                            "seed": run_seed,
                            "pop_size": budget.pop_size,
                            "max_evaluations": budget.max_evaluations,
                            "result_class": cfg.get("result_class", "PILOT"),
                            "feasible_density": geom.density if geom else np.nan,
                            "density_mean_violation": geom.mean_violation if geom else np.nan,
                            "density_median_violation": geom.median_violation if geom else np.nan,
                            "density_n_samples": geom.n_samples if geom else 0,
                        },
                    })

        run_job_batch(
            jobs, fs, budget, workers, rows, done_keys, key_from_row,
            planned, partial_path, log_dir, t0,
            checkpoint_every=50, print_every=10,
            log_prefix=f"{sc.get('label', ds)} ",
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    if os.path.exists(partial_path):
        try:
            os.remove(partial_path)
        except OSError:
            pass
    print(f"\nSaved {len(df)} rows -> {out_path} in {time.time()-t0:.1f}s")
    if len(df):
        print(df.groupby("algorithm")[["FHR", "BestFit", "n_evals", "runtime"]].mean())


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some platforms
    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
