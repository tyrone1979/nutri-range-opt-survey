#!/usr/bin/env python3
"""Precompute Monte Carlo feasible-region density for representative targets (E2 support)."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml

import common as cm
from e1_scaling import load_nhanes_full, sample_food_space
from feasibility_geometry import estimate_feasible_density
from run_benchmark import align_target_to_space, load_representative_targets


from sparsity_ops import prepare_sparsity_target, empirical_meal_caps


def adjust_sparsity(target_array, factor, caps):
    scaled, meta = prepare_sparsity_target(target_array, factor, caps)
    return scaled, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2_density.yaml")
    args = ap.parse_args()
    if os.path.exists(args.config):
        cfg = yaml.safe_load(open(args.config))
    else:
        cfg = {
            "targets_file": "data/representative_targets.csv",
            "targets_total": 12,
            "scale": 10000,
            "density_samples": 2000,
            "sparsity_factors": [0.5, 1.0, 1.5],
            "seed": 42,
            "output": "results/target_analysis/feasible_density.csv",
        }

    matrix, meal_tags, nutrients, _ = cm.load_data(
        "data/nhanes.csv", "data/food_type.csv", "data/target.csv"
    )
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    fs = sample_food_space(full, int(cfg.get("scale", 10000)), seed=int(cfg.get("seed", 42)))

    targets = load_representative_targets(cfg.get("targets_file"), int(cfg.get("targets_total", 12)))
    factors = cfg.get("sparsity_factors", [1.0])
    n_samples = int(cfg.get("density_samples", 2000))
    rows = []

    for t in targets:
        at = align_target_to_space(t, nutrients)
        if at is None:
            continue
        mat = fs.nutrients[:, at.col_idx]
        caps = empirical_meal_caps(
            fs.nutrients, at.col_idx, fs.food_ids, fs.meal_tags,
            fs.min_foods, fs.max_foods, True,
            n_samples=int(cfg.get("cap_samples", 600)),
            percentile=float(cfg.get("cap_percentile", 99.0)),
            seed=int(cfg.get("seed", 42)),
        )
        for fac in factors:
            ta, meta = adjust_sparsity(at.target_array, float(fac), caps)
            geom = estimate_feasible_density(
                mat, fs.meal_tags, fs.food_ids, ta, list(range(at.n_dims)),
                min_foods=fs.min_foods, max_foods=fs.max_foods,
                has_meal_constraint=True,
                n_samples=n_samples,
                seed=int(cfg.get("seed", 42)) + int(1000 * fac),
            )
            rows.append({
                "target_id": at.target_id,
                "scale": fs.n_foods,
                "dim": at.n_dims,
                "sparsity_factor": fac,
                "feasible_density": geom.density,
                "feasible_count": geom.feasible_count,
                "n_samples": geom.n_samples,
                "mean_violation": geom.mean_violation,
                "median_violation": geom.median_violation,
                "p90_violation": geom.p90_violation,
                "meal_feasible_rate": geom.meal_feasible_rate,
                "nutrient_feasible_rate": geom.nutrient_feasible_rate,
                "n_unbounded_capped": meta["n_unbounded_capped"],
            })
            print(f"{at.target_id} ×{fac}: ρ={geom.density:.5f} meanV={geom.mean_violation:.2f} "
                  f"capped={meta['n_unbounded_capped']}")

    out = cfg.get("output", "results/target_analysis/feasible_density.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
