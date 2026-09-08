#!/usr/bin/env python3
"""Watch an E2 log until Saved, then run analyze_e2.py."""
from __future__ import annotations

import argparse
import os
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    print(f"Watching {args.log} for completion...")
    while True:
        done_log = os.path.exists(args.log) and any(
            "Saved" in line and "rows" in line for line in open(args.log, errors="ignore")
        )
        done_csv = os.path.exists(args.csv) and os.path.getsize(args.csv) > 1000
        alive = args.pid is None or os.path.exists(f"/proc/{args.pid}")
        if done_log or (done_csv and not alive):
            print("Detected completion.")
            break
        if args.pid is not None and not alive and not done_csv:
            print("Process died without CSV — abort")
            return 1
        # progress
        if os.path.exists(args.log):
            import re
            ms = re.findall(r"\[(\d+)/(\d{3,})\]", open(args.log, errors="ignore").read())
            if ms:
                a, b = ms[-1]
                print(f"  progress {a}/{b} ({100*int(a)/int(b):.1f}%)", flush=True)
        time.sleep(args.poll)

    cmd = ["python3", "analyze_e2.py", "--input", args.csv, "--tag", args.tag]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
