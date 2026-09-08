#!/usr/bin/env python3
"""Search-dynamics probe: log feasible fraction / violation / fitness vs evaluations."""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

import common as cm
from baseline_algorithms import BMOEA, ACMODiet
from common import compute_max_categories, evaluate_batch
from e1_scaling import load_nhanes_full, sample_food_space
from feasibility_geometry import (
    TrajectoryLogger,
    nutrient_violation,
    snapshot_population_dynamics,
)
from run_benchmark import align_target_to_space, load_representative_targets
from simple_baselines import RandomSearch


def run_bmoea_with_trajectory(fs, target, pop_size, max_evals, seed, out_csv, meta):
    np.random.seed(seed)
    mat = fs.nutrients[:, target.col_idx]
    col_idx = list(range(target.n_dims))
    max_cat = compute_max_categories(fs.food_ids)
    algo = BMOEA(mat, fs.meal_tags, fs.food_ids, max_cat)
    # Manual loop mirroring BMOEA with logging (does not change operators)
    min_foods, max_foods = fs.min_foods, fs.max_foods
    n_valid = len(fs.food_ids)
    population = []
    for _ in range(pop_size):
        n_items = np.random.randint(min_foods, min(max_foods, n_valid) + 1)
        population.append(np.random.choice(fs.food_ids, size=n_items, replace=False))

    logger = TrajectoryLogger()
    n_evals = 0
    gen = 0
    target_array = target.target_array
    crossover_rate, mutation_rate = 0.2, 0.9
    elite_size = 0

    while n_evals < max_evals:
        gen += 1
        scores, feasibles, _, _, _ = evaluate_batch(
            population, mat, target_array, col_idx, fs.meal_tags, max_cat, fs.has_meal_constraint
        )
        n_evals += pop_size
        snap = snapshot_population_dynamics(
            population, mat, target_array, col_idx, fs.meal_tags, max_cat, fs.has_meal_constraint
        )
        # best violation among pop
        viols = [nutrient_violation(ind, mat, target_array, col_idx) for ind in population]
        snap["best_violation"] = float(np.min(viols))
        logger.log(n_evals=n_evals, generation=gen, **snap)

        n_feas = int(feasibles.sum())
        feas_idx = np.where(feasibles)[0]
        num_valid = min(len(feas_idx), pop_size)
        if elite_size < num_valid:
            elite_size = num_valid
        new_pop = [population[i].copy() for i in feas_idx[:num_valid]]
        remaining = pop_size - len(new_pop)
        if remaining > 0:
            infeas_idx = np.where(~feasibles)[0]
            if len(infeas_idx) > 0:
                infeas_scores = np.maximum(scores[infeas_idx], 0)
                s = infeas_scores.sum()
                probs = infeas_scores / s if s > 1e-10 else None
                chosen = np.random.choice(infeas_idx, size=remaining, p=probs, replace=True)
                for idx in chosen:
                    new_pop.append(population[idx].copy())
            elif len(feas_idx) > 0:
                for i in range(remaining):
                    new_pop.append(population[feas_idx[i % len(feas_idx)]].copy())
        population = new_pop[:pop_size]

        for i in range(elite_size, pop_size - 1, 2):
            if np.random.random() < crossover_rate and i + 1 < len(population):
                combined = np.unique(np.concatenate([population[i], population[i + 1]]))
                if len(combined) >= min_foods:
                    n_target = np.random.randint(min_foods, min(max_foods, len(combined)) + 1)
                    population[i] = np.random.choice(combined, size=n_target, replace=False)
        for i in range(len(population)):
            if np.random.random() < mutation_rate:
                indices = population[i]
                action = np.random.randint(0, 3)
                if action == 0 and len(indices) < max_foods:
                    population[i] = np.unique(np.concatenate([indices, [np.random.choice(fs.food_ids)]]))
                elif action == 1 and len(indices) > min_foods:
                    population[i] = np.delete(indices, np.random.randint(0, len(indices)))
                else:
                    pos = np.random.randint(0, len(indices))
                    indices = indices.copy()
                    indices[pos] = np.random.choice(fs.food_ids)
                    population[i] = np.unique(indices)

    logger.save_csv(out_csv, meta=meta)
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--evals", type=int, default=2400)
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    matrix, meal_tags, nutrients, _ = cm.load_data(
        "data/nhanes.csv", "data/food_type.csv", "data/target.csv"
    )
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    fs = sample_food_space(full, args.scale, seed=args.seed)
    targets = load_representative_targets("data/representative_targets.csv", 12)
    raw = targets[args.target_index]
    tgt = align_target_to_space(raw, nutrients)
    assert tgt is not None

    os.makedirs("results/raw/trajectories", exist_ok=True)
    meta = {
        "algorithm": "BMOEA",
        "target_id": tgt.target_id,
        "scale": fs.n_foods,
        "seed": args.seed,
    }
    out = f"results/raw/trajectories/BMOEA_{tgt.target_id}_N{fs.n_foods}.csv"
    run_bmoea_with_trajectory(fs, tgt, args.pop, args.evals, args.seed, out, meta)
    print("Wrote", out)

    # Random baseline trajectory (coarse: log every pop batch)
    np.random.seed(args.seed + 1)
    mat = fs.nutrients[:, tgt.col_idx]
    col_idx = list(range(tgt.n_dims))
    max_cat = compute_max_categories(fs.food_ids)
    logger = TrajectoryLogger()
    best_pop = []
    n_evals = 0
    gen = 0
    from feasibility_geometry import sample_random_subset
    while n_evals < args.evals:
        gen += 1
        batch = [
            sample_random_subset(fs.food_ids, fs.meal_tags, fs.min_foods, fs.max_foods, True)
            for _ in range(min(args.pop, args.evals - n_evals))
        ]
        n_evals += len(batch)
        combined = best_pop + batch
        scores, _, _, _, _ = evaluate_batch(
            combined, mat, tgt.target_array, col_idx, fs.meal_tags, max_cat, True
        )
        order = np.argsort(-scores)[: args.pop]
        best_pop = [combined[i] for i in order]
        snap = snapshot_population_dynamics(
            best_pop, mat, tgt.target_array, col_idx, fs.meal_tags, max_cat, True
        )
        logger.log(n_evals=n_evals, generation=gen, **snap)
    out_r = f"results/raw/trajectories/Random_{tgt.target_id}_N{fs.n_foods}.csv"
    logger.save_csv(out_r, meta={**meta, "algorithm": "Random", "seed": args.seed + 1})
    print("Wrote", out_r)


if __name__ == "__main__":
    main()
