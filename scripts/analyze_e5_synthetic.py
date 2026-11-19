#!/usr/bin/env python3
"""Summarize and plot the corrected E5 synthetic experiment."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "e5_synthetic_5mat.csv"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
STATISTICS = ROOT / "results" / "statistics"


def main() -> None:
    df = pd.read_csv(RAW)
    expected = {"MA", "NSGA2", "SPEA2", "RandomRepair", "Random"}
    df = df[df["algorithm"].isin(expected)].copy()
    if set(df["algorithm"].unique()) != expected:
        raise ValueError("E5 does not contain the complete five-method suite")
    df["inst"] = df["target_id"].str.extract(r"_i(\d+)$")[0].astype(int)
    mat = df.groupby(["scale", "dim", "sparsity", "inst"], as_index=False)["FHR"].mean()

    axes = [
        ("sparsity", [0.5, 1.0, 1.5], "Width factor $f$"),
        ("dim", [6, 12], "Constraint dimension $|J|$"),
        ("scale", [5000, 10000, 25000], "Candidate count $N$"),
    ]
    summary: dict[str, list[dict[str, float]]] = {}
    table_parts = []
    for key, levels, _ in axes:
        part = (
            mat.groupby(key, as_index=False)["FHR"]
            .agg(mean="mean", sd="std", n="count")
            .sort_values(key)
        )
        part.insert(0, "axis", key)
        part = part.rename(columns={key: "level"})
        table_parts.append(part)
        summary[key] = part[["level", "mean", "sd", "n"]].to_dict("records")
        if list(part["level"]) != levels:
            raise ValueError(f"Unexpected E5 levels for {key}: {list(part['level'])}")

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    STATISTICS.mkdir(parents=True, exist_ok=True)
    pd.concat(table_parts, ignore_index=True).to_csv(
        TABLES / "e5_corrected_axis_summary.csv", index=False
    )
    (STATISTICS / "E5_CORRECTED_SUMMARY.json").write_text(
        json.dumps(summary, indent=2)
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
        }
    )
    fig, plot_axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    labels = ["(a) Width", "(b) Dimension", "(c) Scale"]
    for ax, (key, levels, xlabel), title, part in zip(
        plot_axes[:3], axes, labels, table_parts
    ):
        ax.errorbar(
            part["level"],
            part["mean"],
            yerr=part["sd"],
            marker="o",
            color="#1b9e77",
            linewidth=1.2,
            markersize=4,
            capsize=2.5,
            elinewidth=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xticks(levels)
        if key == "scale":
            ax.set_xticklabels(["5K", "10K", "25K"])
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.45)
    ax = plot_axes[3]
    for m, style in ((6, "o-"), (12, "s--")):
        g = mat[mat["dim"] == m].groupby("sparsity")["FHR"].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt=style, color="#7570b3",
                    linewidth=1.2, markersize=4, capsize=2.5, elinewidth=0.8,
                    label=f"$|J|={m}$")
    ax.set_title("(d) Width by dimension")
    ax.set_xlabel("Width factor $f$")
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    plot_axes[0].set_ylabel("Mean FHR")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURES / f"fig_e5_synthetic_fhr.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
