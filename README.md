# NutriOpt-Bench (RCFA)

IEEE TEVC–oriented **experimental survey + benchmark** for **range-constrained food assembly (RCFA)**:
binary subset selection over large food catalogs so nutrient totals fall inside interval boxes
(optional meal coverage). Evolutionary mechanisms are compared under **equal evaluation budgets**.

Nutrition data instantiate the combinatorial regime. This project reports **optimization metrics only**
(FHR, Friedman tests, Kendall τ)—**not** clinical nutrition outcomes and **not** a NutriSwap promotion.

## Authors

**Lei Zhao**<sup>1</sup>, **Fujin Huang**<sup>2</sup>, **Xinning Liu**<sup>3</sup>, **Li Han**<sup>3</sup>, **Ling Kang**<sup>1\*</sup>, **Quan Guo**<sup>1\*</sup>

1. Neusoft Research Institution, Dalian Neusoft University of Information, Dalian, China  
2. Dalian Scientific Test and Control Technology Institute, Dalian, China  
3. Software Department, Dalian Neusoft University of Information, Dalian, China  

\* Corresponding authors: [kangling@neusoft.edu.cn](mailto:kangling@neusoft.edu.cn), [guoquan@neusoft.edu.cn](mailto:guoquan@neusoft.edu.cn)

## Data & code availability

Public repository: [https://github.com/tyrone1979/nutri-range-opt-survey/](https://github.com/tyrone1979/nutri-range-opt-survey/)

PRIMARY CSVs: `results/raw/` · reports: `results/statistics/` · protocol freezes: `docs/`

## Status (PRIMARY complete)

| Track | Status | Artifact |
|-------|--------|----------|
| E1 PRIMARY | **Done** (8400 runs) | `results/raw/e1_scaling_full.csv` |
| E2 PRIMARY | **Done** (3600 runs) | `results/raw/e2_sparsity_full.csv` |
| E3 PRIMARY | **Done** (3840 runs) | `results/raw/e3_dimension_full.csv` |
| E4 meta | **Done** | `results/tables/e4_*.csv`, `docs/` |
| Exact MILP / coverage | Done | `results/raw/exact_small.csv`, `results/target_coverage.csv` |

**Headline (optimization only):** interval sparsity dominates absolute FHR difficulty; NHANES scale alone does not collapse FHR under 100k evaluations; cross-regime Kendall τ remains high (≈0.91–0.98). Recipe catalogs are trivial ceiling regimes and are excluded from ranking. No universal champion is claimed.

**Do not cite** legacy `results/raw/e1_scaling.csv` (DEBUG / unequal budgets).

## Quick start

```bash
pip install -r requirements.txt
python3 validate_data.py
python3 analyze_target_coverage.py
python3 run_exact.py configs/exact_small.yaml

# Fair-budget runners (process pool)
python3 -u run_benchmark.py --config configs/e1_full.yaml --stage main --workers 24
python3 -u run_e2.py --config configs/e2_full.yaml --stage main --workers 32
python3 -u run_e3.py --config configs/e3_full.yaml --stage main --workers 24
```

Workers default to `min(32, ncpu-2)` or `NUTRIOPT_WORKERS` / yaml `workers:`.
Checkpoints write `*.partial.csv` every 50 runs; restart resumes automatically.

## Experiments

| ID | Axis | Config | Runner |
|----|------|--------|--------|
| E1 | Food-space scale | `configs/e1_full.yaml` (~8400 runs) | `run_benchmark.py` |
| E2 | Interval sparsity \{0.5,1.0,1.5\} | `configs/e2_full.yaml` | `run_e2.py` |
| E3 | Nutrient dim \{6,9,12,16\} | `configs/e3_full.yaml` (~3840 runs) | `run_e3.py` |
| E4 | Cross-regime meta | analysis scripts | aggregation only |

Primary protocol: 12 targets × 10 seeds, pop=500, `max_evaluations=100000`, early-stop disabled.

```bash
python3 analyze_e2.py --input results/raw/e2_sparsity_full.csv --tag e2_full
python3 e4_alignment.py
```

## Algorithms

MA, ACMO-Diet, BMOEA, NSGA2, SPEA2, RVEA, CTAEA, Random, Greedy, RandomRepair (+ MILP calibration on reduced pools).  
Mechanism tags: `docs/ALGORITHM_MECHANISM_MATRIX.md`.

## Documentation

| Path | Role |
|------|------|
| `docs/RCFA_FORMULATION.md` | Frozen problem regime |
| `docs/BENCHMARK_PROTOCOL.md` | Fair comparison |
| `docs/FAIRNESS_AUDIT.md` | Budget asymmetries (disclose) |
| `docs/TEVC_CLAIM_EVIDENCE_MATRIX.md` | Claim ↔ evidence |
| `docs/BIBLIOGRAPHY_CLUSTER_MAP.md` | Survey bibliography clusters |
| `docs/BIBLIOGRAPHY_VERIFICATION.json` | DOI schema verification log |
| `REPRODUCIBILITY.md` | Commands + configs |
| `literature/` | Survey protocol + seed matrix |
| `scripts/build_survey_bibliography.py` | Rebuild DOI-verified `.bib` |

## Layout

```
configs/              production YAMLs
run_benchmark.py      E1 fair runner (+ --workers)
run_e2.py / run_e3.py
parallel_jobs.py      process-pool helper
simple_baselines.py / baseline_algorithms.py
sparsity_ops.py       E2 bound sanitization
e4_alignment.py       cross-regime meta
results/raw|tables|figures|statistics|logs
docs/                 protocols + claim matrices
scripts/              bibliography builder
```

## Claims discipline

- Optimization validity ≠ clinical dietary adequacy.
- No overall “best algorithm” / universal champion.
- Cite only `PRIMARY` / labeled `PRIMARY_LITE` / `ROBUSTNESS` / `EXACT`.
- Separate **algorithm search failure** from **intrinsic instance infeasibility**.
- Recipe ceiling regimes must not support ranking claims.
