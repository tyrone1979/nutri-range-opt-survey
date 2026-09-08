#!/usr/bin/env python3
"""
Wait until E1 and E3 PRIMARY finish, then launch E2 full PRIMARY.

Usage:
  python3 -u watch_then_e2_full.py \\
    --e1-pid 1467588 --e3-pid 1468306
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from datetime import datetime


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    return os.path.exists(f"/proc/{pid}")


def last_progress(log_path: str) -> str:
    if not os.path.exists(log_path):
        return "no-log"
    text = open(log_path, errors="ignore").read()
    ms = re.findall(r"\[(\d+)/(\d{3,})\]", text)
    if not ms:
        return "no-progress"
    a, b = ms[-1]
    return f"{a}/{b} ({100 * int(a) / int(b):.1f}%)"


def completed(log_path: str, csv_path: str, pid: int | None) -> tuple[bool, str]:
    """Done if final CSV exists and process is gone (or Saved line present)."""
    csv_ok = os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000
    saved = False
    if os.path.exists(log_path):
        saved = any(
            ("Saved" in line and "rows" in line)
            for line in open(log_path, errors="ignore")
        )
    running = alive(pid)
    if saved and csv_ok:
        return True, "saved+csv"
    if csv_ok and not running:
        return True, "csv+dead"
    if not running and not csv_ok:
        return False, "DEAD_WITHOUT_CSV"
    return False, f"running progress={last_progress(log_path)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-pid", type=int, required=True)
    ap.add_argument("--e3-pid", type=int, required=True)
    ap.add_argument("--e1-log", default="results/logs/e1_full.log")
    ap.add_argument("--e3-log", default="results/logs/e3_full.log")
    ap.add_argument("--e1-csv", default="results/raw/e1_scaling_full.csv")
    ap.add_argument("--e3-csv", default="results/raw/e3_dimension_full.csv")
    ap.add_argument("--e2-config", default="configs/e2_full.yaml")
    ap.add_argument("--e2-log", default="results/logs/e2_full.log")
    ap.add_argument("--e2-csv", default="results/raw/e2_sparsity_full.csv")
    ap.add_argument("--e2-workers", type=int, default=32)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--analyze", action="store_true", default=True)
    args = ap.parse_args()

    print(f"[{datetime.now():%F %T}] Queue: wait E1+E3 → launch E2 full", flush=True)
    print(f"  E1 pid={args.e1_pid}  E3 pid={args.e3_pid}", flush=True)

    e1_fail = e3_fail = False
    while True:
        e1_done, e1_msg = completed(args.e1_log, args.e1_csv, args.e1_pid)
        e3_done, e3_msg = completed(args.e3_log, args.e3_csv, args.e3_pid)
        print(
            f"[{datetime.now():%F %T}] E1={e1_msg} | E3={e3_msg}",
            flush=True,
        )
        if e1_msg == "DEAD_WITHOUT_CSV":
            e1_fail = True
        if e3_msg == "DEAD_WITHOUT_CSV":
            e3_fail = True
        if e1_fail or e3_fail:
            print("Abort: a PRIMARY job died without final CSV.", flush=True)
            return 1
        if e1_done and e3_done:
            break
        time.sleep(args.poll)

    if os.path.exists(args.e2_csv) and os.path.getsize(args.e2_csv) > 1000:
        print(f"E2 full CSV already exists: {args.e2_csv} — skip launch.", flush=True)
    else:
        os.makedirs(os.path.dirname(args.e2_log) or ".", exist_ok=True)
        cmd = [
            "python3", "-u", "run_e2.py",
            "--config", args.e2_config,
            "--stage", "main",
            "--workers", str(args.e2_workers),
        ]
        print(f"[{datetime.now():%F %T}] Launching E2 full: {' '.join(cmd)}", flush=True)
        with open(args.e2_log, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            )
        print(f"E2 full PID={proc.pid} log={args.e2_log}", flush=True)

        # Wait for E2 completion
        while True:
            done, msg = completed(args.e2_log, args.e2_csv, proc.pid)
            print(f"[{datetime.now():%F %T}] E2={msg}", flush=True)
            if msg == "DEAD_WITHOUT_CSV":
                print("E2 died without CSV.", flush=True)
                return 1
            if done:
                break
            time.sleep(args.poll)

    if args.analyze and os.path.exists(args.e2_csv):
        print("Running analyze_e2.py on E2 full...", flush=True)
        r = subprocess.run(
            [
                "python3", "analyze_e2.py",
                "--input", args.e2_csv,
                "--tag", "e2_full",
            ]
        )
        return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
