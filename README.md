# NutriOpt-Bench (RCFA)

IEEE TEVC–oriented **survey + benchmark** for **range-constrained food assembly (RCFA)**:
binary subset selection over large food catalogs so nutrient totals fall inside interval boxes
(optional meal coverage). Evolutionary mechanisms are compared under **equal evaluation budgets**.

Nutrition data instantiate the regime; this is **not** a clinical nutrition paper and **not** a NutriSwap promo.

## Status (2026-09-08)

| Track | Status | Artifact |
|-------|--------|----------|
| E2 PRIMARY_LITE | **Done** (1800 runs, 20k evals × 5 seeds) | `results/raw/e2_sparsity_primary_lite.csv` |
| E1 PRIMARY | **Running** (resume + 24 workers) | `results/raw/e1_scaling_full.csv(.partial)` |
| E3 PRIMARY | **Running** (resume + 24 workers) | `results/raw/e3_dimension_full.csv(.partial)` |
| E2 full PRIMARY | **Queued** after E1+E3 | `watch_then_e2_full.py` → `e2_sparsity_full.csv` |
| Exact MILP / coverage | Done | `results/raw/exact_small.csv`, `results/target_coverage.csv` |
| Manuscript | Rewritten (EC-first draft) | `manuscript/nutriopt_bench.{tex,pdf,docx}` |

**Do not cite** legacy `results/raw/e1_scaling.csv` (DEBUG / unequal budgets).

## Quick start

```bash
pip install -r requirements.txt
python3 validate_data.py
python3 analyze_target_coverage.py
python3 run_exact.py configs/exact_small.yaml

# Fair-budget runners (process pool)
python3 -u run_benchmark.py --config configs/e1_minipilot.yaml --stage pilot --workers 8
python3 -u run_e2.py --config configs/e2_primary_lite.yaml --stage main --workers 16
python3 -u run_e3.py --config configs/e3_full.yaml --stage main --workers 24
```

Workers default to `min(32, ncpu-2)` or `NUTRIOPT_WORKERS` / yaml `workers:`.
Checkpoints write `*.partial.csv` every 50 runs; restart resumes automatically.

### Progress while PRIMARY runs

```bash
grep -E '\[[0-9]+/8400\]' results/logs/e1_full.log | tail -1
grep -E '\[[0-9]+/3840\]' results/logs/e3_full.log | tail -1
tail -f results/logs/watch_then_e2_full.log   # E1+E3 → E2 full queue
```

## Experiments

| ID | Axis | Config | Runner |
|----|------|--------|--------|
| E1 | Food-space scale | `configs/e1_full.yaml` (~8400 runs) | `run_benchmark.py` |
| E2 | Interval sparsity \{0.5,1.0,1.5\} | `e2_primary_lite.yaml` / `e2_full.yaml` | `run_e2.py` |
| E3 | Nutrient dim \{6,9,12,16\} | `configs/e3_full.yaml` (~3840 runs) | `run_e3.py` |
| E4 | Cross-regime meta | `configs/e4_full.yaml` | aggregation only |

Primary protocol: 12 targets × 10 seeds, pop=500, `max_evaluations=100000`, `early_stop_feasible: null`.
PRIMARY_LITE (E2 bridge): 12×5, pop=200, 20k evals.

Analyze E2:

```bash
python3 analyze_e2.py --input results/raw/e2_sparsity_primary_lite.csv --tag e2_primary_lite
```

## Algorithms

BMOEA, ACMO-Diet, MA, NSGA2, RVEA, CTAEA, SPEA2, Random, Greedy, RandomRepair (+ MILP calibration on reduced pools).
Mechanism tags: see `docs/ALGORITHM_MECHANISM_MATRIX.md`.

## Documentation

| Path | Role |
|------|------|
| `docs/RCFA_FORMULATION.md` | Frozen problem regime |
| `docs/BENCHMARK_PROTOCOL.md` | Fair comparison |
| `docs/FAIRNESS_AUDIT.md` | Budget asymmetries (disclose) |
| `docs/E2_SPARSITY_PROTOCOL.md` | Sanitized sparsity scaling |
| `docs/TEVC_RECONSTRUCTION_AUDIT.md` | Phase II readiness |
| `docs/CLAIM_EVIDENCE_MATRIX.md` | What may be claimed |
| `REPRODUCIBILITY.md` | Commands + configs |
| `literature/` | Survey protocol + seed matrix |
| `manuscript/` | TEVC-oriented draft |

## Layout

```
configs/           production YAMLs
run_benchmark.py   E1 fair runner (+ --workers)
run_e2.py / run_e3.py
parallel_jobs.py   process-pool helper
simple_baselines.py / baseline_algorithms.py
sparsity_ops.py    E2 bound sanitization
results/raw|tables|figures|statistics|logs
```

## Claims discipline

- Optimization validity ≠ clinical dietary adequacy.
- No overall “best algorithm” until full PRIMARY E1–E3 finish.
- Cite only `PRIMARY` / `PRIMARY_LITE` (labeled) / `ROBUSTNESS` / `EXACT`.
- RandomRepair under-used budget on E2 PRIMARY_LITE (fixed in code; re-run before hard ranking).
