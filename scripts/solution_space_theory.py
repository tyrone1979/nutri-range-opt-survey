#!/usr/bin/env python3
"""Solution-space quantities that do not depend on a particular solver.

inversion  Random draws i.i.d. from the initializer D, so one run of B draws hits F
           with probability 1-(1-p_F)^B.  Inverting the Random hit counts gives
           p_F on every E1/E2/E3 cell with exact binomial intervals, and the
           random-search-equivalent budget multiplier of every other solver.
mc         Direct Monte Carlo of p_F on the E2 boxes (same construction as run_e2.py),
           with face-wise and per-coordinate pass frequencies.

Usage:
  python scripts/solution_space_theory.py inversion
  python scripts/solution_space_theory.py mc --draws 1e8 --workers 60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy.stats import beta

TABLES = ROOT / "results" / "tables"
STATS = ROOT / "results" / "statistics"
SUPP = ROOT / "manuscript" / "swevo" / "supplement_tables"

AXES = {
    "E1": ("results/raw/swevo_e1_pymoo_corrected.csv", "scale"),
    "E2": ("results/raw/e2_sparsity_full.csv", "sparsity"),
    "E3": ("results/raw/swevo_e3_pymoo_corrected.csv", "dim"),
}
SOLVERS = ["MA", "NSGA2", "SPEA2", "Greedy", "RandomRepair", "Random"]


def miss_to_p(miss: float, budget: float) -> float:
    """Per-draw probability p with (1-p)^B = miss."""
    if miss <= 0.0:
        return math.inf
    if miss >= 1.0:
        return 0.0
    return -math.log(miss) / budget


def cp_interval(h: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if h == 0 else float(beta.ppf(alpha / 2, h, n - h + 1))
    hi = 1.0 if h == n else float(beta.ppf(1 - alpha / 2, h + 1, n - h))
    return lo, hi


def load_axis(axis: str) -> pd.DataFrame:
    path, level = AXES[axis]
    df = pd.read_csv(path)
    df = df[(df["status"] == "ok") & df["algorithm"].isin(SOLVERS)].copy()
    df["level"] = df[level]
    df["hit"] = (df["FHR"] > 0).astype(int)
    return df


def inversion_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_target, per_cell = [], []
    for axis in AXES:
        df = load_axis(axis)
        rnd = df[df["algorithm"] == "Random"]
        budget = float(rnd["n_evals"].median())
        for level, g in rnd.groupby("level"):
            ps = []
            for tid, gt in g.groupby("target_id"):
                h, n = int(gt["hit"].sum()), int(len(gt))
                lo, hi = cp_interval(h, n)
                p = miss_to_p(1 - h / n, budget)
                ps.append(p)
                per_target.append({
                    "axis": axis, "level": level, "target_id": tid,
                    "hits": h, "runs": n, "budget": budget,
                    "p_hat": p,
                    "p_lo": miss_to_p(1 - lo, budget),
                    "p_hi": miss_to_p(1 - hi, budget),
                    "censored": "zero" if h == 0 else ("all" if h == n else ""),
                })
            H, R = int(g["hit"].sum()), int(len(g))
            lo, hi = cp_interval(H, R)
            ps = np.sort(np.asarray(ps))
            per_cell.append({
                "axis": axis, "level": level,
                "hits": H, "runs": R, "draws": R * budget,
                "run_hit_rate": H / R,
                "p_pooled": miss_to_p(1 - H / R, budget),
                "p_pooled_lo": miss_to_p(1 - lo, budget),
                "p_pooled_hi": miss_to_p(1 - hi, budget),
                "p_median_target": float(np.median(ps)),
                "targets_zero_hits": int(sum(1 for p in ps if p == 0.0)),
                "targets_all_hits": int(sum(1 for p in ps if math.isinf(p))),
                "n_targets": int(len(ps)),
            })
    return pd.DataFrame(per_target), pd.DataFrame(per_cell)


def kappa(pi_a: float, pi_r: float, n_a: int, n_r: int) -> tuple[float, str]:
    """log P(miss | A) / log P(miss | Random) with continuity bounds at 0 and 1."""
    flag = ""
    if pi_a >= 1.0:
        pi_a, flag = (n_a - 0.5) / n_a, ">="
    if pi_r <= 0.0:
        pi_r, flag = 0.5 / n_r, ">="
    if pi_r >= 1.0:
        pi_r, flag = (n_r - 0.5) / n_r, "<="
    if pi_a <= 0.0:
        return 0.0, flag
    return math.log1p(-pi_a) / math.log1p(-pi_r), flag


def amplification_table(n_boot: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for axis in AXES:
        df = load_axis(axis)
        agg = df.groupby(["level", "algorithm", "target_id"])["hit"].agg(["sum", "count"]).reset_index()
        for level, g in agg.groupby("level"):
            targets = sorted(g["target_id"].unique())
            tab = {a: g[g["algorithm"] == a].set_index("target_id").reindex(targets) for a in g["algorithm"].unique()}
            r = tab["Random"]
            for alg, t in tab.items():
                if alg == "Random":
                    continue
                pi_a = t["sum"].sum() / t["count"].sum()
                pi_r = r["sum"].sum() / r["count"].sum()
                k, flag = kappa(pi_a, pi_r, int(t["count"].sum()), int(r["count"].sum()))
                boots = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(targets), len(targets))
                    ta, tr = t.iloc[idx], r.iloc[idx]
                    kb, _ = kappa(ta["sum"].sum() / ta["count"].sum(), tr["sum"].sum() / tr["count"].sum(),
                                  int(ta["count"].sum()), int(tr["count"].sum()))
                    boots.append(kb)
                rows.append({
                    "axis": axis, "level": level, "algorithm": alg,
                    "fhr": pi_a, "fhr_random": pi_r,
                    "kappa": k, "kappa_flag": flag,
                    "kappa_lo": float(np.percentile(boots, 2.5)),
                    "kappa_hi": float(np.percentile(boots, 97.5)),
                })
    return pd.DataFrame(rows)


def run_inversion() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    per_target, per_cell = inversion_tables()
    amp = amplification_table()
    per_target.to_csv(TABLES / "solution_space_random_inversion_by_target.csv", index=False)
    per_cell.to_csv(TABLES / "solution_space_random_inversion_by_cell.csv", index=False)
    amp.to_csv(TABLES / "solution_space_amplification.csv", index=False)
    pd.set_option("display.width", 200)
    print(per_cell.to_string(index=False))
    print(amp.to_string(index=False))


# ---------------------------------------------------------------- Monte Carlo

_G: dict = {}


def build_e2_instances(seed: int = 42) -> dict:
    import common as cm
    from e1_scaling import load_nhanes_full, sample_food_space
    from run_benchmark import align_target_to_space, load_representative_targets
    from sparsity_ops import empirical_meal_caps, prepare_sparsity_target

    matrix, meal_tags, nutrients, _ = cm.load_data("data/nhanes.csv", "data/food_type.csv", "data/target.csv")
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    fs = sample_food_space(full, 10000, seed=seed)
    assert np.array_equal(fs.food_ids, np.arange(fs.n_foods))

    targets = []
    for t in load_representative_targets("data/representative_targets.csv", 12):
        at = align_target_to_space(t, nutrients)
        if at is not None and at.n_dims >= 4:
            targets.append(at)
    targets = targets[:12]

    boxes = []
    for tgt in targets:
        caps = empirical_meal_caps(
            fs.nutrients, tgt.col_idx, fs.food_ids, fs.meal_tags,
            fs.min_foods, fs.max_foods, fs.has_meal_constraint,
            n_samples=600, percentile=99.0, seed=seed + tgt.n_dims,
        )
        for fac in (0.5, 1.0, 1.5):
            arr, _ = prepare_sparsity_target(tgt.target_array, fac, caps)
            boxes.append({
                "target_id": tgt.target_id, "factor": fac,
                "col_idx": np.asarray(tgt.col_idx), "lo": arr[:, 0].astype(np.float64),
                "hi": arr[:, 1].astype(np.float64),
            })
    lower_extra = max(0, int(fs.min_foods) - 3)
    upper_extra = max(lower_extra, min(int(fs.max_foods), fs.n_foods) - 3)
    return {
        "nutrients": np.asarray(fs.nutrients, dtype=np.float32),
        "meal_pools": [np.flatnonzero(fs.meal_tags == m) for m in (0, 1, 2)],
        "n": int(fs.n_foods), "extra": (lower_extra, upper_extra),
        "boxes": boxes,
    }


def _draw(rng: np.random.Generator, c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """c i.i.d. subsets from common.random_individual: one item per meal class plus
    n_extra distinct further items, n_extra uniform on [lower_extra, upper_extra]."""
    n = _G["n"]
    lo_e, hi_e = _G["extra"]
    base = np.stack([p[rng.integers(0, len(p), c)] for p in _G["meal_pools"]], axis=1)
    k = rng.integers(lo_e, hi_e + 1, c)
    mask = np.arange(hi_e)[None, :] < k[:, None]
    extra = rng.integers(0, n, (c, hi_e))
    sentinel = -1 - np.arange(hi_e)[None, :]
    todo = np.arange(c)
    while todo.size:
        comb = np.concatenate([base[todo], np.where(mask[todo], extra[todo], sentinel)], axis=1)
        comb.sort(axis=1)
        bad = todo[(comb[:, 1:] == comb[:, :-1]).any(axis=1)]
        if bad.size:
            extra[bad] = rng.integers(0, n, (bad.size, hi_e))
        todo = bad
    return base, np.where(mask, extra, n), k


def _mc_task(args):
    box_i, n_draws, seed = args
    b = _G["boxes"][box_i]
    mat = _G["padded"][box_i]
    lo, hi = b["lo"], b["hi"]
    rng = np.random.default_rng(seed)
    d = lo.size
    out = {"draws": 0, "hits": 0, "lower_ok": 0, "upper_ok": 0,
           "coord_ok": np.zeros(d, dtype=np.int64)}
    chunk = 20000
    left = n_draws
    while left > 0:
        c = min(chunk, left)
        base, extra, _ = _draw(rng, c)
        z = mat[base].sum(axis=1, dtype=np.float32) + mat[extra].sum(axis=1, dtype=np.float32)
        below = np.maximum(0.0, lo - z)
        above = np.maximum(0.0, z - hi)
        feas = (below + above).sum(axis=1) < 1e-6
        lok = (below <= 0).all(axis=1)
        uok = (above <= 0).all(axis=1)
        out["draws"] += c
        out["hits"] += int(feas.sum())
        out["lower_ok"] += int(lok.sum())
        out["upper_ok"] += int(uok.sum())
        out["coord_ok"] += ((below <= 0) & (above <= 0)).sum(axis=0)
        left -= c
    return box_i, out


def run_mc(n_draws: int, workers: int, factors: list[float], task_size: int) -> None:
    global _G
    t0 = time.time()
    _G = build_e2_instances()
    pad = np.zeros((1, _G["nutrients"].shape[1]), dtype=np.float32)
    full = np.vstack([_G["nutrients"], pad])
    _G["padded"] = [full[:, b["col_idx"]] for b in _G["boxes"]]
    sel = [i for i, b in enumerate(_G["boxes"]) if b["factor"] in factors]
    tasks = []
    for i in sel:
        left, j = n_draws, 0
        while left > 0:
            m = min(task_size, left)
            tasks.append((i, m, 10_000_019 * (i + 1) + j))
            left -= m
            j += 1
    print(f"{len(sel)} boxes x {n_draws:.3g} draws in {len(tasks)} tasks on {workers} workers", flush=True)
    acc: dict[int, dict] = {}
    with Pool(workers) as pool:
        for k, (i, o) in enumerate(pool.imap_unordered(_mc_task, tasks), 1):
            a = acc.setdefault(i, {"draws": 0, "hits": 0, "lower_ok": 0, "upper_ok": 0,
                                   "coord_ok": np.zeros_like(o["coord_ok"])})
            for key in ("draws", "hits", "lower_ok", "upper_ok", "coord_ok"):
                a[key] = a[key] + o[key]
            if k % max(1, len(tasks) // 20) == 0:
                print(f"  {k}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)

    rows = []
    for i, a in sorted(acc.items()):
        b = _G["boxes"][i]
        n, h = a["draws"], a["hits"]
        lo, hi = cp_interval(h, n)
        marg = a["coord_ok"] / n
        p_ind = float(np.prod(marg))
        rows.append({
            "target_id": b["target_id"], "factor": b["factor"], "n_coords": int(b["lo"].size),
            "draws": n, "hits": h, "p_hat": h / n, "p_lo": lo, "p_hi": hi,
            "p_lower_faces": a["lower_ok"] / n, "p_upper_faces": a["upper_ok"] / n,
            "p_independent": p_ind,
            "coupling_ratio": (h / n) / p_ind if p_ind > 0 and h > 0 else float("nan"),
            "min_marginal": float(marg.min()),
        })
    df = pd.DataFrame(rows)
    name = "solution_space_mc_e2.csv"
    out = TABLES / name
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old["factor"].isin(factors)]
        df = pd.concat([old, df], ignore_index=True).sort_values(["factor", "target_id"])
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    print(f"Saved {out} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- cross-axis collapse

EAS = ["MA", "NSGA2", "SPEA2"]


def _rate(pi: float, n: int) -> float:
    """log of -ln(1-pi), the log effective per-run hazard, with continuity at 0 and 1."""
    pi = min(max(pi, 0.5 / n), (n - 0.5) / n)
    return math.log(-math.log1p(-pi))


def run_collapse() -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    cell = pd.read_csv(TABLES / "solution_space_random_inversion_by_cell.csv")
    rows = []
    for axis in AXES:
        df = load_axis(axis)
        for level, g in df.groupby("level"):
            ea = g[g["algorithm"].isin(EAS)]
            pf = cell[(cell["axis"] == axis) & (cell["level"] == level)].iloc[0]
            r = {"axis": axis, "level": level, "p_F": pf["p_pooled"],
                 "fhr_ea": ea["hit"].mean(), "n_ea": len(ea)}
            for a in EAS + ["Random"]:
                r[f"fhr_{a}"] = g[g["algorithm"] == a]["hit"].mean()
            rows.append(r)
    c = pd.DataFrame(rows)
    c["x"] = np.log10(c["p_F"])
    c["y"] = [_rate(p, n) for p, n in zip(c["fhr_ea"], c["n_ea"])]

    def fit(d):
        b, a = np.polyfit(d["x"], d["y"], 1)
        return a, b

    def predict(a, b, x):
        return 1 - np.exp(-np.exp(a + b * x))

    a, b = fit(c)
    pred = predict(a, b, c["x"])
    loao = []
    for axis in AXES:
        tr, te = c[c["axis"] != axis], c[c["axis"] == axis]
        aa, bb = fit(tr)
        err = np.abs(predict(aa, bb, te["x"]) - te["fhr_ea"])
        loao.append({"held_out": axis, "slope": bb, "mae": float(err.mean()), "max_err": float(err.max())})
    rho, p_rho = spearmanr(c["p_F"], c["fhr_ea"])
    res = {
        "slope": b, "intercept": a,
        "r2": float(1 - np.sum((c["y"] - (a + b * c["x"])) ** 2) / np.sum((c["y"] - c["y"].mean()) ** 2)),
        "mae_in_sample": float(np.mean(np.abs(pred - c["fhr_ea"]))),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "leave_one_axis_out": loao,
    }
    c.to_csv(TABLES / "solution_space_collapse_cells.csv", index=False)
    (STATS / "SOLUTION_SPACE_COLLAPSE.json").write_text(json.dumps(res, indent=2))
    print(c[["axis", "level", "p_F", "fhr_ea", "fhr_MA", "fhr_NSGA2", "fhr_SPEA2", "fhr_Random"]].to_string(index=False))
    print(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    marker = {"E1": "o", "E2": "s", "E3": "^"}
    label = {"E1": "scale (E1)", "E2": "width (E2)", "E3": "dimension (E3)"}
    for axis, g in c.groupby("axis"):
        ax.scatter(g["p_F"], g["fhr_ea"], marker=marker[axis], s=42, color="#1b6ca8",
                   label=f"EA mean, {label[axis]}", zorder=3)
        ax.scatter(g["p_F"], g["fhr_Random"], marker=marker[axis], s=42, facecolors="none",
                   edgecolors="#b2182b", zorder=3)
    e5_path = ROOT / "results" / "raw" / "e5_synthetic_5mat.csv"
    if e5_path.exists():
        e5 = pd.read_csv(e5_path)
        e5 = e5[e5["algorithm"] != "BMOEA"].copy()
        e5["hit"] = (e5["FHR"] > 0).astype(int)
        e5_rows = []
        for (N, m, f), g in e5.groupby(["scale", "dim", "sparsity"]):
            r = g[g["algorithm"] == "Random"]
            ea = g[g["algorithm"].isin(EAS)]
            h, n = int(r["hit"].sum()), len(r)
            if 0 < h < n:
                pf = float(-np.log(1 - h / n) / 1e5)
                e5_rows.append({"p_F": pf, "fhr_ea": float(ea["hit"].mean())})
        if e5_rows:
            e5d = pd.DataFrame(e5_rows)
            ax.scatter(e5d["p_F"], e5d["fhr_ea"], marker="D", s=30, color="#636363",
                       alpha=0.85, label="EA mean, synthetic (E5, uncensored)", zorder=2)
    grid = np.logspace(np.log10(c["p_F"].min()) - 0.3, np.log10(c["p_F"].max()) + 0.3, 200)
    ax.plot(grid, predict(a, b, np.log10(grid)), color="#1b6ca8", lw=1.2, label="fitted EA curve (E1\u2013E3)")
    budget = 1e5
    ax.plot(grid, 1 - np.exp(-grid * budget), color="#b2182b", lw=1.0, ls="--",
            label=r"Random, $1-(1-\bar p_F)^B$")
    ax.set_xscale("log")
    ax.set_xlabel(r"feasible frequency $\bar p_F$ under reference sampling $D$")
    ax.set_ylabel("feasible hit rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_pf_collapse.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out / "fig_pf_collapse.pdf")


# ---------------------------------------------------------------- summary

AXIS_LABEL = {"E1": "Scale", "E2": "Width", "E3": "Dimension"}
SOLVER_LABEL = {"MA": "MA", "NSGA2": "NSGA-II", "SPEA2": "SPEA2", "Greedy": "Greedy", "RandomRepair": "RandomRepair"}


def _level(axis: str, level: float) -> str:
    if axis == "E1":
        return f"{int(level):,}".replace(",", "{,}")
    if axis == "E2":
        return f"$f={level:g}$"
    return f"$|J|={int(level)}$"


def write_inversion_tables(cell: pd.DataFrame, amp: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Axis & Level & Hits/runs & $\hat p_F^{\mathrm{RS}}$ & 95\% interval & Targets 0 / all hits \\",
        r"\midrule",
    ]
    for r in cell.itertuples(index=False):
        lines.append(
            f"{AXIS_LABEL[r.axis]} & {_level(r.axis, r.level)} & {r.hits}/{r.runs} & {_sci(r.p_pooled)} & "
            f"[{_sci(r.p_pooled_lo)}, {_sci(r.p_pooled_hi)}] & {r.targets_zero_hits} / {r.targets_all_hits} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (SUPP / "table_s14_pf_inversion.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{longtable}{@{}llccc@{}}",
        r"\caption{Random-search-equivalent budget $\kappa_A$ by cell and solver. "
        r"Interval is the 95\% bootstrap interval over targets; $\geq$ marks a cell where the solver hit on every run "
        r"or Random never hit, so the point value is a bound.}",
        r"\label{tab:s15}\\",
        r"\toprule",
        r"Axis & Level & Solver & FHR & $\kappa_A$ (95\% interval) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Axis & Level & Solver & FHR & $\kappa_A$ (95\% interval) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    order = {a: i for i, a in enumerate(SOLVER_LABEL)}
    amp = amp.assign(o=amp["algorithm"].map(order)).sort_values(["axis", "level", "o"])
    for r in amp.itertuples(index=False):
        flag = r"$\geq$\," if r.kappa_flag == ">=" else (r"$\leq$\," if r.kappa_flag == "<=" else "")
        lines.append(
            f"{AXIS_LABEL[r.axis]} & {_level(r.axis, r.level)} & {SOLVER_LABEL[r.algorithm]} & {r.fhr:.3f} & "
            f"{flag}{r.kappa:.1f} ({r.kappa_lo:.1f}--{r.kappa_hi:.1f}) \\\\"
        )
    lines.append(r"\end{longtable}")
    (SUPP / "table_s15_kappa.tex").write_text("\n".join(lines) + "\n")
    print("Saved S14/S15 tables")


def write_summary() -> None:
    cell = pd.read_csv(TABLES / "solution_space_random_inversion_by_cell.csv")
    amp = pd.read_csv(TABLES / "solution_space_amplification.csv")
    SUPP.mkdir(parents=True, exist_ok=True)
    write_inversion_tables(cell, amp)
    summary = {"inversion": cell.to_dict("records"), "amplification": amp.to_dict("records")}
    mc_path = TABLES / "solution_space_mc_e2.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        g = mc.groupby("factor").agg(
            draws=("draws", "first"),
            mean_p=("p_hat", "mean"), median_p=("p_hat", "median"),
            min_p=("p_hat", "min"), max_p=("p_hat", "max"),
            zero_targets=("hits", lambda s: int((s == 0).sum())),
            mean_p_lower=("p_lower_faces", "mean"), mean_p_upper=("p_upper_faces", "mean"),
            median_coupling=("coupling_ratio", "median"),
        ).reset_index()
        summary["mc_e2_by_factor"] = g.to_dict("records")
        print(g.to_string(index=False))
        tgt = pd.read_csv(TABLES / "solution_space_random_inversion_by_target.csv")
        tgt = tgt[tgt["axis"] == "E2"].rename(columns={"level": "factor"})
        m = mc.merge(tgt, on=["target_id", "factor"], suffixes=("_mc", "_inv"))
        m["expected_run_hit"] = 1 - np.exp(-m["p_hat_mc"] * m["budget"])
        m["consistent"] = [
            cp_interval(int(h), int(r))[0] <= e <= cp_interval(int(h), int(r))[1]
            for h, r, e in zip(m["hits_inv"], m["runs"], m["expected_run_hit"])
        ]
        summary["mc_vs_inversion_consistent"] = int(m["consistent"].sum())
        summary["mc_vs_inversion_total"] = int(len(m))
        m.to_csv(TABLES / "solution_space_mc_vs_inversion.csv", index=False)
        print(f"MC vs Random-inversion consistent on {m['consistent'].sum()}/{len(m)} cells")
        write_supplement_table(mc)
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / "SOLUTION_SPACE_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=float))


def _sci(x: float) -> str:
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return rf"${m:.1f}\times10^{{{e}}}$"


def write_supplement_table(mc: pd.DataFrame) -> None:
    lines = [
        r"\begin{longtable}{@{}lcccccc@{}}",
        r"\caption{Direct Monte Carlo estimate of the feasible frequency $p_F$ on the E2 boxes. "
        r"Interval is the exact 95\% binomial interval; lower and upper are the frequencies with which all "
        r"lower faces, respectively all upper faces, are met.}",
        r"\label{tab:s13}\\",
        r"\toprule",
        r"Target & $f$ & Draws & $\hat p_F$ & 95\% interval & Lower & Upper \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Direct Monte Carlo estimate of $p_F$ (continued).}\\",
        r"\toprule",
        r"Target & $f$ & Draws & $\hat p_F$ & 95\% interval & Lower & Upper \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{\emph{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for r in mc.sort_values(["target_id", "factor"]).itertuples(index=False):
        tid = str(r.target_id).replace("_", r"\_")
        lines.append(
            f"{tid} & {r.factor:g} & {_sci(r.draws)} & {_sci(r.p_hat)} & "
            f"[{_sci(r.p_lo)}, {_sci(r.p_hi)}] & {r.p_lower_faces:.3f} & {r.p_upper_faces:.3f} \\\\"
        )
    lines.append(r"\end{longtable}")
    SUPP.mkdir(parents=True, exist_ok=True)
    (SUPP / "table_s13_pf_mc.tex").write_text("\n".join(lines) + "\n")
    print("Saved", SUPP / "table_s13_pf_mc.tex")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inversion", "mc", "summary", "collapse"])
    ap.add_argument("--draws", type=float, default=1e7)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--task-size", type=float, default=2e6)
    args = ap.parse_args()
    if args.mode == "inversion":
        run_inversion()
        write_summary()
    elif args.mode == "mc":
        run_mc(int(args.draws), args.workers, args.factors, int(args.task_size))
        write_summary()
    elif args.mode == "collapse":
        run_collapse()
    else:
        write_summary()


if __name__ == "__main__":
    main()
