#!/usr/bin/env python3
"""Watch budget 50k completion, stop full rerun, inject E2@100k, launch 200k-only."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARTIAL = ROOT / "results/raw/e_budget_f05.csv.partial.csv"
LOG = ROOT / "results/logs/budget_watch.log"


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def pids():
    out = subprocess.check_output(["pgrep", "-f", "run_budget_sensitivity.py"] , text=True)
    return [int(x) for x in out.split() if x.strip()]


def main():
    log("watching for 50k completion (720 rows)...")
    while True:
        if not PARTIAL.exists():
            time.sleep(30)
            continue
        df = pd.read_csv(PARTIAL)
        n50 = int((df["max_evaluations"] == 50000).sum())
        n25 = int((df["max_evaluations"] == 25000).sum())
        log(f"rows={len(df)} n25={n25} n50={n50}")
        if n50 >= 720:
            break
        time.sleep(60)

    log("50k complete — stopping budget workers to skip redundant 100k recompute")
    for pid in pids():
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"SIGTERM {pid}")
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in pids():
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"SIGKILL {pid}")
        except ProcessLookupError:
            pass

    # merge 25k+50k + E2 100k
    subprocess.check_call(["python3", str(ROOT / "scripts/merge_budget_with_e2_100k.py")])

    # prepare partial for 200k-only resume: keep 25/50/100, drop any incomplete 200k
    merged = pd.read_csv(ROOT / "results/raw/e_budget_f05.csv")
    merged = merged[merged["max_evaluations"].isin([25000, 50000, 100000])]
    # write as partial so runner resumes only missing 200k if config lists it
    # Better: write dedicated 200k config
    cfg = (ROOT / "configs/e_budget_200k.yaml")
    cfg.write_text(
        """
experiment: e_budget
result_class: BUDGET_SENSITIVITY
seed: 42
population_size: 500
max_generations: 10000
early_stop_feasible: null
targets_total: 12
repeats: 10
targets_file: data/representative_targets.csv
sparsity_factor: 0.5
cap_samples: 600
cap_percentile: 99.0
evaluation_budgets: [200000]
algorithms:
  - MA
  - ACMO-Diet
  - BMOEA
  - NSGA2
  - RandomRepair
  - Random
scales:
  - dataset: nhanes_sample
    size: 10000
    label: "NHANES-10K"
output_dir: results/raw
log_dir: results/logs
output_name: e_budget_200k.csv
workers: 16
""".lstrip()
    )
    # seed partial empty for 200k
    out200 = ROOT / "results/raw/e_budget_200k.csv.partial.csv"
    if out200.exists():
        out200.unlink()
    log("launching 200k-only budget run")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.Popen(
        ["python3", "run_budget_sensitivity.py", "--config", "configs/e_budget_200k.yaml", "--workers", "16"],
        cwd=str(ROOT),
        env=env,
        stdout=open(ROOT / "results/logs/e_budget_200k.log", "w"),
        stderr=subprocess.STDOUT,
    )
    log("done; 200k running in background")


if __name__ == "__main__":
    main()
