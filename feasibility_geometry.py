"""
Feasibility geometry and search-dynamics instrumentation for NutriOpt-Bench (TEVC Phase II).

Provides:
- Monte Carlo feasible-region density estimates
- Distance-to-feasibility proxies (nutrient violation)
- Trajectory logger (per-generation snapshots)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from common import evaluate_batch, compute_max_categories, random_individual


def sample_random_subset(all_valid_idx, meal_tags, min_foods, max_foods, has_meal_constraint):
    if has_meal_constraint:
        ind = random_individual(
            all_valid_idx, meal_tags, min_foods=min_foods, max_foods=max_foods
        )
        if ind is not None and min_foods <= len(ind) <= max_foods:
            return np.asarray(ind, dtype=np.int32)
    n = int(np.random.randint(min_foods, min(max_foods, len(all_valid_idx)) + 1))
    return np.random.choice(all_valid_idx, size=n, replace=False).astype(np.int32)


def nutrient_violation(ind, matrix, target_array, col_idx):
    total = matrix[ind][:, col_idx].sum(axis=0)
    lo, hi = target_array[:, 0], target_array[:, 1]
    return float((np.maximum(0, lo - total) + np.maximum(0, total - hi)).sum())


@dataclass
class FeasibilityGeometry:
    n_samples: int
    feasible_count: int
    density: float
    mean_violation: float
    median_violation: float
    p90_violation: float
    meal_feasible_rate: float
    nutrient_feasible_rate: float


def estimate_feasible_density(
    matrix: np.ndarray,
    meal_tags: np.ndarray,
    all_valid_idx: np.ndarray,
    target_array: np.ndarray,
    col_idx: Sequence[int],
    min_foods: int,
    max_foods: int,
    has_meal_constraint: bool = True,
    n_samples: int = 2000,
    seed: Optional[int] = None,
) -> FeasibilityGeometry:
    """
    Monte Carlo estimate of feasible-region density under the same sampling
    distribution used by RandomSearch initialization (not volume of {0,1}^N).
    """
    if seed is not None:
        np.random.seed(seed)
    max_cat = compute_max_categories(all_valid_idx)
    feas = 0
    meal_ok = 0
    nut_ok = 0
    viols = []
    pop = []
    for _ in range(n_samples):
        ind = sample_random_subset(all_valid_idx, meal_tags, min_foods, max_foods, has_meal_constraint)
        pop.append(ind)
        v = nutrient_violation(ind, matrix, target_array, col_idx)
        viols.append(v)
        if v < 1e-6:
            nut_ok += 1
        if has_meal_constraint:
            meals = meal_tags[ind]
            if (np.any(meals == 0) and np.any(meals == 1) and np.any(meals == 2)):
                meal_ok += 1
        else:
            meal_ok += 1

    scores, feasibles, _, _, _ = evaluate_batch(
        pop, matrix, target_array, col_idx, meal_tags, max_cat, has_meal_constraint
    )
    feas = int(feasibles.sum())
    viols = np.asarray(viols, dtype=float)
    return FeasibilityGeometry(
        n_samples=n_samples,
        feasible_count=feas,
        density=feas / max(n_samples, 1),
        mean_violation=float(viols.mean()),
        median_violation=float(np.median(viols)),
        p90_violation=float(np.percentile(viols, 90)),
        meal_feasible_rate=meal_ok / max(n_samples, 1),
        nutrient_feasible_rate=nut_ok / max(n_samples, 1),
    )


@dataclass
class TrajectoryLogger:
    """Collect per-generation search dynamics."""
    records: List[Dict] = field(default_factory=list)

    def log(
        self,
        *,
        n_evals: int,
        generation: int,
        frac_feasible: float,
        best_violation: float,
        best_fitness: float,
        mean_diversity: float,
        n_feasible: int,
    ) -> None:
        self.records.append({
            "n_evals": int(n_evals),
            "generation": int(generation),
            "frac_feasible": float(frac_feasible),
            "best_violation": float(best_violation),
            "best_fitness": float(best_fitness),
            "mean_diversity": float(mean_diversity),
            "n_feasible": int(n_feasible),
        })

    def save_csv(self, path: str, meta: Optional[Dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "n_evals", "generation", "frac_feasible", "best_violation",
            "best_fitness", "mean_diversity", "n_feasible",
        ]
        if meta:
            fieldnames = list(meta.keys()) + fieldnames
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.records:
                row = dict(meta or {})
                row.update(r)
                w.writerow(row)


def snapshot_population_dynamics(
    population,
    matrix,
    target_array,
    col_idx,
    meal_tags,
    max_categories,
    has_meal_constraint,
) -> Dict[str, float]:
    scores, feasibles, diversities, _, _ = evaluate_batch(
        population, matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint
    )
    viols = [
        nutrient_violation(ind, matrix, target_array, col_idx) for ind in population
    ]
    n_feas = int(feasibles.sum())
    return {
        "frac_feasible": float(feasibles.mean()) if len(feasibles) else 0.0,
        "best_violation": float(np.min(viols)) if viols else float("inf"),
        "best_fitness": float(scores.max()) if len(scores) else 0.0,
        "mean_diversity": float(diversities[feasibles].mean()) if n_feas > 0 else 0.0,
        "n_feasible": n_feas,
    }
