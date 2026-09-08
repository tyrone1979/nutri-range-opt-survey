#!/usr/bin/env python3
"""Run MILP exact baseline on small instances."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import yaml

from exact_milp import solve_rcda_milp
from run_benchmark import align_target_to_space, load_representative_targets
from e1_scaling import load_nhanes_full, sample_food_space
import common as cm


def main(config_path="configs/exact_small.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    matrix, meal_tags, nutrients, _ = cm.load_data("data/nhanes.csv", "data/food_type.csv", "data/target.csv")
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    fs = sample_food_space(full, int(cfg["scales"][0]["size"]), seed=cfg.get("seed", 42))

    targets = load_representative_targets(cfg["targets_file"], int(cfg["targets_total"]))
    rows = []
    for t in targets:
        at = align_target_to_space(t, nutrients)
        if at is None:
            continue
        # build matrix view
        mat = fs.nutrients[:, at.col_idx]
        res = solve_rcda_milp(
            mat, at.target_array, list(range(at.n_dims)),
            meal_tags=fs.meal_tags,
            min_foods=cfg.get("min_foods", 5),
            max_foods=cfg.get("max_foods", 15),
            has_meal_constraint=cfg.get("has_meal_constraint", True),
            time_limit=cfg.get("time_limit", 60),
            candidate_limit=cfg.get("candidate_limit", 300),
            seed=cfg.get("seed", 42),
        )
        rows.append({
            "target_id": at.target_id,
            "dim": at.n_dims,
            "scale": fs.n_foods,
            "candidate_limit": cfg.get("candidate_limit", 300),
            **{k: v for k, v in res.items() if k != "selected"},
        })
        print(at.target_id, res["status"], "feasible=", res["feasible"], "viol=", res["violation"], f"t={res['runtime']:.2f}s")

    out = os.path.join(cfg.get("output_dir", "results/raw"), cfg.get("output_name", "exact_small.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
