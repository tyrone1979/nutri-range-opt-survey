#!/usr/bin/env python3
"""Controlled MA mechanism ablation on E2 tight cell (f=0.5, NHANES-10K).

Variants (identical encoding, fitness, selection, crossover, local search,
seeds, targets, pop=500, max_evals=100,000):
  - MA_no_dir_del : uniform random deletion instead of center-directed deletion
  - MA_no_dir_mut : uniform random add, delete, and replace (no center guidance)
  - MA_meal_init  : full MA, but initialized from meal-seeded distribution D
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as cm
from baseline_algorithms import MemeticAlgorithm, _safe_card_randint, evaluate_single
from benchmark_core import make_seed, set_global_seed
from common import compute_max_categories, evaluate_batch
from e1_scaling import Target, load_nhanes_full, sample_food_space
from run_benchmark import align_target_to_space, load_representative_targets
from simple_baselines import _rand_ind
from sparsity_ops import empirical_meal_caps, prepare_sparsity_target


class AblatedMA(MemeticAlgorithm):
    def __init__(self, matrix, meal_tags, all_valid_idx, max_categories,
                 directed_delete=True, directed_add_replace=True, meal_init=False):
        super().__init__(matrix, meal_tags, all_valid_idx, max_categories)
        self.directed_delete = directed_delete
        self.directed_add_replace = directed_add_replace
        self.meal_init = meal_init

    def run(self, target_array, col_idx, pop_size=500, n_generations=10000,
            max_evals=100000, min_foods=5, max_foods=30, has_meal_constraint=True,
            early_stop_feasible=None):
        n_valid = len(self.all_valid_idx)
        population = []
        for _ in range(pop_size):
            if self.meal_init:
                selected = _rand_ind(self.all_valid_idx, self.meal_tags,
                                     min_foods, max_foods, has_meal_constraint)
            else:
                n_items = _safe_card_randint(min_foods, max_foods, n_valid)
                selected = np.random.choice(self.all_valid_idx, size=n_items, replace=False)
            population.append(selected)

        ffg = None
        ffg_evals = None
        feasible_found = False
        n_evals = 0
        best_feasible = None
        best_fitness = 0.0
        target_center = (target_array[:, 0] + target_array[:, 1]) / 2.0
        all_valid_nutrients = self.matrix[self.all_valid_idx][:, col_idx]

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break
            scores, feasibles, diversities, n_foods_arr, contrib_diversities = evaluate_batch(
                population, self.matrix, target_array, col_idx,
                self.meal_tags, self.max_categories, has_meal_constraint)
            n_evals += pop_size

            n_feas = feasibles.sum()
            if n_feas > 0:
                feas_scores = scores[feasibles]
                best_local_idx = np.argmax(feas_scores)
                best_local_score = feas_scores[best_local_idx]
                if best_local_score > best_fitness:
                    best_fitness = best_local_score
                    feas_indices = np.where(feasibles)[0]
                    best_feasible = population[feas_indices[best_local_idx]].copy()
                if not feasible_found:
                    ffg = gen + 1
                    ffg_evals = n_evals
                    feasible_found = True

            if n_feas > 0 and gen % 3 == 0:
                n_ls = min(5, n_feas)
                for idx in np.where(feasibles)[0][:n_ls]:
                    improved = self._local_search(
                        population[idx], target_array, col_idx,
                        target_center, all_valid_nutrients, max_iter=10, min_foods=min_foods)
                    fit, _, feas, _, _ = evaluate_single(
                        improved, self.matrix, target_array, col_idx,
                        self.meal_tags, self.max_categories, has_meal_constraint)
                    n_evals += 1
                    if feas and fit > best_fitness:
                        best_fitness = fit
                        best_feasible = improved.copy()
                        population[idx] = improved
                        scores[idx] = fit
                        feasibles[idx] = True

            feas_idx = np.where(feasibles)[0]
            num_valid = min(len(feas_idx), pop_size)
            new_pop = [population[i].copy() for i in feas_idx[:num_valid]]
            elite_size = num_valid

            remaining = pop_size - len(new_pop)
            if remaining > 0:
                infeas_idx = np.where(~feasibles)[0]
                if len(infeas_idx) > 0:
                    infeas_scores = np.maximum(scores[infeas_idx], 0)
                    s = infeas_scores.sum()
                    if s > 1e-10:
                        probs = infeas_scores / s
                        chosen = np.random.choice(infeas_idx, size=remaining, p=probs, replace=True)
                    else:
                        chosen = np.random.choice(infeas_idx, size=remaining, replace=True)
                    for idx in chosen:
                        new_pop.append(population[idx].copy())
                else:
                    for i in range(remaining):
                        idx = feas_idx[i % len(feas_idx)]
                        new_pop.append(population[idx].copy())

            population = new_pop[:pop_size]

            for i in range(elite_size, pop_size - 1, 2):
                if np.random.random() < 0.3:
                    combined = np.unique(np.concatenate([population[i], population[i + 1]]))
                    n_target = _safe_card_randint(min_foods, max_foods, len(combined))
                    population[i] = np.random.choice(combined, size=n_target, replace=False)

            for i in range(pop_size):
                if np.random.random() < 0.7:
                    indices = population[i]
                    if len(indices) == 0:
                        continue
                    action = np.random.randint(0, 3)
                    if action == 0 and len(indices) < max_foods:
                        if self.directed_add_replace and np.random.random() < 0.3 and len(indices) >= min_foods:
                            current_total = self.matrix[indices][:, col_idx].sum(axis=0)
                            candidates = np.random.choice(self.all_valid_idx, size=min(10, n_valid), replace=False)
                            best_score = -float('inf')
                            best_cand = None
                            for cand in candidates:
                                new_total = current_total + self.matrix[cand][col_idx]
                                score = -np.sum((new_total - target_center) ** 2)
                                if score > best_score:
                                    best_score = score
                                    best_cand = cand
                            if best_cand is not None:
                                indices = np.concatenate([indices, [best_cand]])
                        else:
                            new_idx = np.random.choice(self.all_valid_idx)
                            indices = np.concatenate([indices, [new_idx]])
                        population[i] = np.unique(indices)

                    elif action == 1 and len(indices) > min_foods:
                        if self.directed_delete:
                            current_total = self.matrix[indices][:, col_idx].sum(axis=0)
                            best_score = float('inf')
                            worst_pos = 0
                            for j, food in enumerate(indices):
                                without = current_total - self.matrix[food][col_idx]
                                score = np.sum((without - target_center) ** 2)
                                if score < best_score:
                                    best_score = score
                                    worst_pos = j
                            population[i] = np.delete(indices, worst_pos)
                        else:
                            pos = np.random.randint(0, len(indices))
                            population[i] = np.delete(indices, pos)

                    else:
                        pos = np.random.randint(0, len(indices))
                        old_food = indices[pos]
                        if self.directed_add_replace and np.random.random() < 0.3 and len(indices) >= min_foods:
                            current_total = self.matrix[indices][:, col_idx].sum(axis=0)
                            without = current_total - self.matrix[old_food][col_idx]
                            candidates = np.random.choice(self.all_valid_idx, size=min(10, n_valid), replace=False)
                            best_score = float('inf')
                            best_cand = old_food
                            for cand in candidates:
                                new_total = without + self.matrix[cand][col_idx]
                                score = np.sum((new_total - target_center) ** 2)
                                if score < best_score:
                                    best_score = score
                                    best_cand = cand
                            indices[pos] = best_cand
                        else:
                            indices[pos] = np.random.choice(self.all_valid_idx)
                        population[i] = np.unique(indices)

        final_scores, final_feasibles, final_diversities, final_n_foods, contrib_diversities = evaluate_batch(
            population, self.matrix, target_array, col_idx,
            self.meal_tags, self.max_categories, has_meal_constraint)
        n_evals += pop_size

        if final_feasibles.sum() == 0 and best_feasible is not None:
            population[0] = best_feasible
            final_scores, final_feasibles, final_diversities, final_n_foods, contrib_diversities = evaluate_batch(
                population, self.matrix, target_array, col_idx,
                self.meal_tags, self.max_categories, has_meal_constraint)

        return {
            "FHR": 1.0 if np.any(final_feasibles) else 0.0,
            "FFG_evals": ffg_evals if ffg_evals else n_evals,
            "n_evals": n_evals,
            "BestFit": float(final_scores.max()),
        }


_G_FS = None
_G_TARGETS = None


def _worker(job):
    variant, tid, rep, run_seed, flags = job
    tgt = _G_TARGETS[tid]
    fs = _G_FS
    set_global_seed(run_seed)
    mat = fs.nutrients[:, tgt.col_idx]
    col_idx = list(range(len(tgt.nutrient_names)))
    max_cat = compute_max_categories(fs.food_ids)
    t0 = time.time()
    out = AblatedMA(mat, fs.meal_tags, fs.food_ids, max_cat, **flags).run(
        tgt.target_array, col_idx, pop_size=500, n_generations=10000,
        max_evals=100000, min_foods=fs.min_foods, max_foods=fs.max_foods,
        has_meal_constraint=fs.has_meal_constraint)
    out.update({
        "algorithm": variant,
        "target_id": tid,
        "rep": rep,
        "seed": run_seed,
        "sparsity": 0.5,
        "runtime": time.time() - t0,
    })
    return out


def main():
    global _G_FS, _G_TARGETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="results/raw/ma_ablation_f05.csv")
    args = ap.parse_args()

    matrix, meal_tags, nutrients, _ = cm.load_data(
        "data/nhanes.csv", "data/food_type.csv", "data/target.csv")
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    _G_FS = sample_food_space(full, 10000, seed=42)

    raw_targets = load_representative_targets("data/representative_targets.csv", 12)
    _G_TARGETS = {}
    for t in raw_targets:
        at = align_target_to_space(t, nutrients)
        if at is None or at.n_dims < 4:
            continue
        caps = empirical_meal_caps(
            _G_FS.nutrients, at.col_idx, _G_FS.food_ids, _G_FS.meal_tags,
            _G_FS.min_foods, _G_FS.max_foods, _G_FS.has_meal_constraint,
            n_samples=600, percentile=99.0, seed=42 + at.n_dims)
        scaled_arr, _ = prepare_sparsity_target(at.target_array, 0.5, caps)
        _G_TARGETS[at.target_id] = Target(
            at.target_id, scaled_arr[:, 0], scaled_arr[:, 1],
            at.nutrient_names, at.col_idx, scaled_arr)

    variants = [
        ("MA_no_dir_del", dict(directed_delete=False, directed_add_replace=True, meal_init=False)),
        ("MA_no_dir_mut", dict(directed_delete=False, directed_add_replace=False, meal_init=False)),
        ("MA_meal_init",  dict(directed_delete=True,  directed_add_replace=True,  meal_init=True)),
    ]
    jobs = []
    for variant, flags in variants:
        for tid in sorted(_G_TARGETS):
            for rep in range(10):
                run_seed = make_seed(42, "e2_sparsity_f0.5", 10000, tid, "MA", rep)
                jobs.append((variant, tid, rep, run_seed, flags))

    t0 = time.time()
    print(f"Running {len(jobs)} ablation jobs on {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(_worker, jobs))
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out} in {time.time()-t0:.1f}s")

    # Compare against primary MA
    e2 = pd.read_csv("results/raw/e2_sparsity_full.csv")
    ma = e2[(e2.sparsity == 0.5) & (e2.algorithm == "MA")].copy()
    print("\nSummary (FHR mean +/- seed SD):")
    for name, g in [("MA (primary)", ma)] + [(v, df[df.algorithm == v]) for v, _ in variants]:
        sm = g.groupby("rep")["FHR"].mean()
        tgt_cov = (g.groupby("target_id")["FHR"].mean() > 0).sum()
        print(f"  {name:15s}: FHR={g['FHR'].mean():.3f} +/- {sm.std(ddof=1):.3f}  targets_hit={tgt_cov}/12")
    x = ma.groupby("target_id")["FHR"].mean()
    print("\nPaired Wilcoxon vs primary MA (12 target blocks):")
    for v, _ in variants:
        y = df[df.algorithm == v].groupby("target_id")["FHR"].mean()
        xa, ya = x.align(y, join="inner")
        diff = ya - xa
        p = float(wilcoxon(xa, ya).pvalue) if (diff != 0).any() else 1.0
        print(f"  {v:15s}: diff={diff.mean():+.3f}  wilcoxon_p={p:.5f}")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
