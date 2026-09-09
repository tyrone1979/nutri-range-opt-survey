#!/usr/bin/env python3
"""Expanded exact feasibility calibration across reduced pool sizes."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml

import common as cm
from e1_scaling import load_nhanes_full, sample_food_space
from exact_milp import solve_rcda_milp
from run_benchmark import align_target_to_space, load_representative_targets


def classify_result(res: dict) -> str:
    """Map soft-slack CBC outcomes to calibration labels A–D vocabulary."""
    status = str(res.get("status", ""))
    feasible = bool(res.get("feasible", False))
    if feasible:
        return "proven_feasible"
    # Optimal soft-slack with positive violation ⇒ certified empty under the reduced pool
    if status == "Optimal" and not feasible:
        return "proven_infeasible"
    sl = status.lower()
    if "infeas" in sl:
        return "proven_infeasible"
    if any(tok in sl for tok in ("not solved", "undefined", "error", "unavailable", "time")):
        return "unresolved"
    return "unresolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exact_multi_pool.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(cfg.get("output_dir", "results/raw"), cfg.get("output_name", "exact_multi_pool.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    matrix, meal_tags, nutrients, _ = cm.load_data(
        "data/nhanes.csv", "data/food_type.csv", "data/target.csv"
    )
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)

    targets = []
    for t in load_representative_targets(cfg.get("targets_file"), int(cfg.get("targets_total", 12))):
        at = align_target_to_space(t, nutrients)
        if at is not None:
            targets.append(at)

    pools = [int(x) for x in cfg.get("pool_sizes", [250, 500, 1000])]
    cand = int(cfg.get("candidate_limit", 300))
    time_limit = int(cfg.get("time_limit", 60))
    seed = int(cfg.get("seed", 42))
    rows = []
    for ps in pools:
        fs = sample_food_space(full, ps, seed=seed)
        for tgt in targets:
            mat = fs.nutrients[:, tgt.col_idx]
            res = solve_rcda_milp(
                mat, tgt.target_array, list(range(tgt.n_dims)),
                meal_tags=fs.meal_tags,
                min_foods=int(cfg.get("min_foods", 5)),
                max_foods=int(cfg.get("max_foods", 15)),
                has_meal_constraint=bool(cfg.get("has_meal_constraint", True)),
                time_limit=time_limit,
                candidate_limit=min(cand, fs.n_foods),
                seed=seed,
            )
            label = classify_result(res)
            rows.append({
                "target_id": tgt.target_id,
                "dim": tgt.n_dims,
                "pool_size": ps,
                "candidate_limit": min(cand, fs.n_foods),
                "calibration_label": label,
                **{k: v for k, v in res.items() if k != "selected"},
            })
            print(
                f"pool={ps} {tgt.target_id}: {label} status={res.get('status')} "
                f"viol={res.get('violation')} t={res.get('runtime', 0):.2f}s"
            )
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("Saved", out)
    print(df.groupby(["pool_size", "calibration_label"]).size())


if __name__ == "__main__":
    main()
