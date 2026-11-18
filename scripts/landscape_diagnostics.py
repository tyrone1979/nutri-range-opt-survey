#!/usr/bin/env python3
"""E2 solution-space diagnostics: box vs feasible frequency, face-wise violation, one-step descent.

Does not rerun E1--E5. Samples the same initialization distribution used by Random.
"""
from __future__ import annotations

import math
import os
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import common as cm
from e1_scaling import load_nhanes_full, sample_food_space
from estimate_feasibility import adjust_sparsity
from feasibility_geometry import sample_random_subset
from run_benchmark import align_target_to_space, load_representative_targets
from sparsity_ops import empirical_meal_caps


def log10_decision_space(n: int, kmin: int, kmax: int) -> float:
    logs = []
    for k in range(int(kmin), int(kmax) + 1):
        if k < 0 or k > n:
            continue
        logs.append(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    if not logs:
        return float("nan")
    m = max(logs)
    return (m + math.log(sum(math.exp(x - m) for x in logs))) / math.log(10)


def parts(total: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float, float]:
    lower = np.maximum(0.0, lo - total)
    upper = np.maximum(0.0, total - hi)
    return float(lower.sum()), float(upper.sum()), float(lower.sum() + upper.sum())


def one_step_rates(
    ind: np.ndarray,
    matrix: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    pool: np.ndarray,
    rng: np.random.Generator,
    n_add: int,
    n_del: int,
    n_swap: int,
) -> dict:
    ind = np.unique(ind.astype(np.int32))
    base = matrix[ind].sum(axis=0)
    bl, bu, bv = parts(base, lo, hi)
    outside = np.setdiff1d(pool, ind, assume_unique=False)

    def rate(candidates: np.ndarray, mode: str) -> tuple[int, int]:
        if candidates.size == 0:
            return 0, 0
        if mode == "add":
            totals = base + matrix[candidates]
        elif mode == "del":
            totals = base - matrix[candidates]
        else:
            # candidates is (n, 2): drop col0, add col1
            totals = base - matrix[candidates[:, 0]] + matrix[candidates[:, 1]]
        lower = np.maximum(0.0, lo - totals)
        upper = np.maximum(0.0, totals - hi)
        viol = lower.sum(axis=1) + upper.sum(axis=1)
        return int(np.sum(viol < bv - 1e-8)), int(candidates.shape[0])

    add_idx = rng.choice(outside, size=min(n_add, outside.size), replace=False) if outside.size else np.array([], dtype=np.int32)
    del_idx = rng.choice(ind, size=min(n_del, ind.size), replace=False) if ind.size else np.array([], dtype=np.int32)
    n_sw = min(n_swap, ind.size, outside.size)
    if n_sw:
        swap = np.column_stack([
            rng.choice(ind, size=n_sw, replace=True),
            rng.choice(outside, size=n_sw, replace=True),
        ])
    else:
        swap = np.zeros((0, 2), dtype=np.int32)
    a_hit, a_n = rate(add_idx, "add")
    d_hit, d_n = rate(del_idx, "del")
    s_hit, s_n = rate(swap, "swap")
    return {
        "base_lower": bl,
        "base_upper": bu,
        "base_violation": bv,
        "add_improve": a_hit,
        "add_trials": a_n,
        "del_improve": d_hit,
        "del_trials": d_n,
        "swap_improve": s_hit,
        "swap_trials": s_n,
    }


def load_cfg(path: str) -> dict:
    cfg = {
        "targets_file": "data/representative_targets.csv",
        "targets_total": 12,
        "scale": 10000,
        "sparsity_factors": [0.5, 1.0, 1.5],
        "seed": 42,
        "cap_samples": 600,
        "cap_percentile": 99.0,
    }
    if not os.path.exists(path):
        return cfg
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if val.startswith("["):
            cfg[key] = [float(x) for x in val.strip("[]").split(",") if x.strip()]
        else:
            try:
                cfg[key] = int(val)
            except ValueError:
                try:
                    cfg[key] = float(val)
                except ValueError:
                    cfg[key] = val.strip('"')
    return cfg


def write_swevo_table(df: pd.DataFrame) -> None:
    """Write the per-target diagnostic table used by the SWEO supplement."""
    lines = [
        r"\begin{longtable}{@{}lccccccc@{}}",
        r"\caption{Per-target initialization geometry on the width axis "
        r"($n=1200$ draws). A zero frequency is censored.}",
        r"\label{tab:s11}\\",
        r"\toprule",
        r"Target & $f$ & $\hat p_F$ & Lower & Upper & Add & Delete & Swap \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Per-target initialization geometry (continued).}\\",
        r"\toprule",
        r"Target & $f$ & $\hat p_F$ & Lower & Upper & Add & Delete & Swap \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{\emph{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in df.itertuples(index=False):
        target_id = str(row.target_id).replace("_", r"\_")
        p_f = "0" if int(row.feasible_count) == 0 else f"{row.feasible_frequency:.4f}"
        lines.append(
            f"{target_id} & {row.sparsity_factor:g} & {p_f} & "
            f"{row.mean_lower:.0f} & {row.mean_upper:.0f} & "
            f"{row.add_improve_rate:.2f} & {row.del_improve_rate:.2f} & "
            f"{row.swap_improve_rate:.2f} \\\\"
        )
    lines.append(r"\end{longtable}")
    out = Path("manuscript/swevo/supplement_tables/table_s11_landscape.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("Saved", out)


def main():
    cfg = load_cfg("configs/e2_density.yaml")
    n_samples = int(os.environ.get("LANDSCAPE_DENSITY_N", "4000"))
    n_bases = int(os.environ.get("LANDSCAPE_BASES", "12"))
    seed0 = int(cfg.get("seed", 42))

    matrix, meal_tags, nutrients, _ = cm.load_data(
        "data/nhanes.csv", "data/food_type.csv", "data/target.csv"
    )
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    fs = sample_food_space(full, int(cfg.get("scale", 10000)), seed=seed0)
    log10_x = log10_decision_space(int(fs.n_foods), int(fs.min_foods), int(fs.max_foods))

    targets = load_representative_targets(cfg.get("targets_file"), int(cfg.get("targets_total", 12)))
    factors = [float(f) for f in cfg.get("sparsity_factors", [0.5, 1.0, 1.5])]
    rows = []
    for t in targets:
        at = align_target_to_space(t, nutrients)
        if at is None:
            continue
        mat = np.asarray(fs.nutrients[:, at.col_idx], dtype=np.float64)
        caps = empirical_meal_caps(
            fs.nutrients, at.col_idx, fs.food_ids, fs.meal_tags,
            fs.min_foods, fs.max_foods, True,
            n_samples=int(cfg.get("cap_samples", 600)),
            percentile=float(cfg.get("cap_percentile", 99.0)),
            seed=seed0,
        )
        for fac in factors:
            ta, meta = adjust_sparsity(at.target_array, fac, caps)
            lo, hi = ta[:, 0].astype(np.float64), ta[:, 1].astype(np.float64)
            target_hash = int.from_bytes(
                hashlib.sha256(str(at.target_id).encode("utf-8")).digest()[:4],
                byteorder="big",
            )
            rng = np.random.default_rng(
                (seed0 + int(1000 * fac) + target_hash) % (2**32)
            )
            np.random.seed(seed0 + int(1000 * fac))
            lowers, uppers, viols = [], [], []
            feas = 0
            bases = []
            for s in range(n_samples):
                ind = sample_random_subset(
                    fs.food_ids, fs.meal_tags, fs.min_foods, fs.max_foods, True
                )
                total = mat[ind].sum(axis=0)
                lo_v, up_v, v = parts(total, lo, hi)
                lowers.append(lo_v)
                uppers.append(up_v)
                viols.append(v)
                if v < 1e-6:
                    feas += 1
                elif len(bases) < n_bases:
                    bases.append(ind)
            acc = {k: 0 for k in (
                "add_improve", "add_trials", "del_improve", "del_trials", "swap_improve", "swap_trials"
            )}
            for ind in bases:
                stats = one_step_rates(ind, mat, lo, hi, fs.food_ids, rng, 20, 8, 16)
                for k in acc:
                    acc[k] += stats[k]
            def frac(h, n):
                return float(h) / float(n) if n else float("nan")
            rows.append({
                "target_id": at.target_id,
                "scale": int(fs.n_foods),
                "kmin": int(fs.min_foods),
                "kmax": int(fs.max_foods),
                "log10_decision_space": log10_x,
                "sparsity_factor": fac,
                "n_samples": n_samples,
                "feasible_count": feas,
                "feasible_frequency": feas / n_samples,
                "censored": int(feas == 0),
                "mean_lower": float(np.mean(lowers)),
                "mean_upper": float(np.mean(uppers)),
                "mean_violation": float(np.mean(viols)),
                "median_violation": float(np.median(viols)),
                "add_improve_rate": frac(acc["add_improve"], acc["add_trials"]),
                "del_improve_rate": frac(acc["del_improve"], acc["del_trials"]),
                "swap_improve_rate": frac(acc["swap_improve"], acc["swap_trials"]),
                "one_step_improve_rate": frac(
                    acc["add_improve"] + acc["del_improve"] + acc["swap_improve"],
                    acc["add_trials"] + acc["del_trials"] + acc["swap_trials"],
                ),
                "n_unbounded_capped": meta["n_unbounded_capped"],
            })
            print(
                f"{at.target_id} f={fac}: pF={feas/n_samples:.5f} "
                f"V={np.mean(viols):.1f} step={rows[-1]['one_step_improve_rate']:.3f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/statistics", exist_ok=True)
    out = "results/tables/landscape_e2_by_target.csv"
    df.to_csv(out, index=False)
    summary = (
        df.groupby("sparsity_factor", as_index=False)
        .agg(
            mean_pF=("feasible_frequency", "mean"),
            frac_censored=("censored", "mean"),
            mean_lower=("mean_lower", "mean"),
            mean_upper=("mean_upper", "mean"),
            mean_violation=("mean_violation", "mean"),
            add_improve_rate=("add_improve_rate", "mean"),
            del_improve_rate=("del_improve_rate", "mean"),
            swap_improve_rate=("swap_improve_rate", "mean"),
            one_step_improve_rate=("one_step_improve_rate", "mean"),
            log10_decision_space=("log10_decision_space", "first"),
            n_samples=("n_samples", "first"),
        )
    )
    summary.to_csv("results/tables/landscape_e2_by_factor.csv", index=False)
    summary.to_json("results/statistics/LANDSCAPE_E2_SUMMARY.json", orient="records", indent=2)
    write_swevo_table(df)
    print(summary.to_string(index=False))
    print("Saved", out)


if __name__ == "__main__":
    main()
