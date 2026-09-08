"""
Sparsity / interval-width operators for NutriOpt-Bench E2.

Fixes pathological center-scaling when upper bounds are sentinels
(e.g. 99999, 9999999) representing one-sided / effectively unbounded constraints.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

# Treat U >= this as "effectively unbounded" sentinel
UNBOUNDED_THRESHOLD = 1.0e5


def is_unbounded(upper: np.ndarray, threshold: float = UNBOUNDED_THRESHOLD) -> np.ndarray:
    return np.asarray(upper, dtype=float) >= threshold


def empirical_meal_caps(
    matrix: np.ndarray,
    col_idx: Sequence[int],
    all_valid_idx: np.ndarray,
    meal_tags: np.ndarray,
    min_foods: int,
    max_foods: int,
    has_meal_constraint: bool = True,
    n_samples: int = 800,
    percentile: float = 99.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Estimate per-nutrient high totals under random feasible-cardinality meals.
    Used to replace sentinel upper bounds before width scaling.
    """
    rng = np.random.default_rng(seed)
    m = len(col_idx)
    totals = np.zeros((n_samples, m), dtype=float)
    for s in range(n_samples):
        if has_meal_constraint:
            chosen = []
            for meal in (0, 1, 2):
                pool = all_valid_idx[meal_tags[all_valid_idx] == meal]
                if len(pool) == 0:
                    pool = all_valid_idx
                chosen.append(int(rng.choice(pool)))
            n_extra = int(rng.integers(0, max(1, max_foods - 2)))
            if n_extra > 0:
                chosen.extend(rng.choice(all_valid_idx, size=n_extra, replace=True).tolist())
            ind = np.unique(np.asarray(chosen, dtype=np.int32))
            if len(ind) < min_foods:
                extra = rng.choice(all_valid_idx, size=min_foods - len(ind), replace=False)
                ind = np.unique(np.concatenate([ind, extra]))
            if len(ind) > max_foods:
                ind = rng.choice(ind, size=max_foods, replace=False)
        else:
            n = int(rng.integers(min_foods, max_foods + 1))
            ind = rng.choice(all_valid_idx, size=n, replace=False)
        totals[s] = matrix[ind][:, list(col_idx)].sum(axis=0)
    caps = np.percentile(totals, percentile, axis=0)
    # Ensure strictly positive width allowance
    return np.maximum(caps, 1e-6)


def sanitize_target_array(
    target_array: np.ndarray,
    caps: np.ndarray,
    threshold: float = UNBOUNDED_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace sentinel uppers with empirical caps (at least L+eps).
    Returns (sanitized_array, unbounded_mask).
    """
    lo = target_array[:, 0].astype(np.float64).copy()
    hi = target_array[:, 1].astype(np.float64).copy()
    mask = is_unbounded(hi, threshold)
    hi_san = hi.copy()
    hi_san[mask] = np.maximum(caps[mask], lo[mask] + 1.0)
    # Also clamp absurd finite but huge values already below threshold? no
    out = np.column_stack([lo, hi_san]).astype(np.float32)
    return out, mask


def adjust_sparsity_array(target_array: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale interval half-width about center by `factor`.
    Requires finite (sanitized) bounds. factor<1 tightens, >1 loosens.
    """
    lo = target_array[:, 0].astype(np.float64)
    hi = target_array[:, 1].astype(np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * float(factor)
    new_lo = np.maximum(center - half, 0.0)
    new_hi = np.maximum(center + half, new_lo + 1e-6)
    return np.column_stack([new_lo, new_hi]).astype(np.float32)


def prepare_sparsity_target(
    target_array: np.ndarray,
    factor: float,
    caps: np.ndarray,
    threshold: float = UNBOUNDED_THRESHOLD,
) -> Tuple[np.ndarray, dict]:
    """
    Sanitize sentinel uppers, then apply sparsity.

    Finite intervals: scale half-width about center by `factor`.
    Formerly-unbounded dimensions (one-sided):
      - Keep U at empirical cap (does not invent a tight box about cap/2).
      - If L > 0 (minimum nutrient, e.g. fiber): L' = L / factor
        (factor<1 raises minimum → tighter; factor>1 lowers minimum → looser).
      - If L == 0 (no meaningful lower bound): keep L=0; optional
        U' = cap * factor for loosening upper pressure only when factor>1,
        and U' = cap * factor when factor<1 to tighten the soft cap.
    """
    lo = target_array[:, 0].astype(np.float64).copy()
    hi = target_array[:, 1].astype(np.float64).copy()
    mask = is_unbounded(hi, threshold)
    factor = float(factor)

    new_lo = lo.copy()
    new_hi = hi.copy()

    # Finite dimensions: classical center scaling
    finite = ~mask
    if np.any(finite):
        c = 0.5 * (lo[finite] + hi[finite])
        half = 0.5 * (hi[finite] - lo[finite]) * factor
        new_lo[finite] = np.maximum(c - half, 0.0)
        new_hi[finite] = np.maximum(c + half, new_lo[finite] + 1e-6)

    # One-sided / sentinel dimensions
    if np.any(mask):
        cap = np.maximum(caps[mask], lo[mask] + 1.0)
        L = lo[mask]
        # Soft upper from empirical cap, scaled mildly with factor
        U = np.maximum(cap * factor, L + 1.0)
        # Minimum-style lower bounds
        L2 = L.copy()
        pos = L > 1e-8
        L2[pos] = np.maximum(L[pos] / factor, 0.0)
        new_lo[mask] = L2
        new_hi[mask] = np.maximum(U, L2 + 1e-6)

    out = np.column_stack([new_lo, new_hi]).astype(np.float32)
    meta = {
        "n_unbounded_capped": int(mask.sum()),
        "unbounded_dims": np.where(mask)[0].tolist(),
        "sparsity_factor": factor,
        "sanitize_threshold": float(threshold),
        "rule": "finite_center_scale+onesided_L_div_factor",
    }
    return out, meta


def adjust_target_object(target, factor: float, caps: np.ndarray, threshold: float = UNBOUNDED_THRESHOLD):
    """Return a new Target-like object if dataclass; else (array, meta)."""
    scaled, meta = prepare_sparsity_target(target.target_array, factor, caps, threshold)
    return type(target)(
        target_id=target.target_id,
        lower_bounds=scaled[:, 0],
        upper_bounds=scaled[:, 1],
        nutrient_names=target.nutrient_names,
        col_idx=target.col_idx,
        target_array=scaled,
    ), meta
