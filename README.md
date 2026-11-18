# NutriOpt-Bench (RCFA)

Empirical **benchmark** for **range-constrained food assembly (RCFA)**: binary subset selection over large food catalogs so nutrient totals fall inside interval boxes (optional meal coverage). Evolutionary and baseline solvers are compared under a shared hard feasibility predicate and **equal evaluation budgets**.

Nutrition data instantiate the combinatorial regime. This repository reports **optimization metrics only** (feasible hit rate / FHR, Friedman tests, Kendall τ)—**not** clinical nutrition outcomes.

The current primary suite uses a **corrected decoder/initializer** (duplicate slots collapsed; random subsets respect cardinality without replacement) and a six-solver core. An empirical manuscript draft targeting *Swarm and Evolutionary Computation* lives on the full project mirror; this public GitHub tree focuses on code and result artifacts.

## Authors

**Lei Zhao**, **Ling Kang**\*, **Xinning Liu**, **Li Han**, **Quan Guo**\*, **Fujin Huang**

1. Neusoft Research Institution, Dalian Neusoft University of Information, Dalian, China  
2. Dalian Scientific Test and Control Technology Institute, Dalian, China  
3. Software Department, Dalian Neusoft University of Information, Dalian, China  

\* Corresponding authors: [kangling@neusoft.edu.cn](mailto:kangling@neusoft.edu.cn), [guoquan@neusoft.edu.cn](mailto:guoquan@neusoft.edu.cn)

## Data & code availability

- GitHub (code + results): [https://github.com/tyrone1979/nutri-range-opt-survey/](https://github.com/tyrone1979/nutri-range-opt-survey/)
- Gitee (full project mirror): [https://gitee.com/sacourse/nutri-range-opt-survey/](https://gitee.com/sacourse/nutri-range-opt-survey/)

Primary CSVs: `results/raw/` · summaries: `results/statistics/` · figures: `results/figures/` and `figs/` · protocols: `docs/`

## Status (corrected primary)

| Track | Status | Artifact |
|-------|--------|----------|
| E1 scale (corrected) | **Done** (4320 runs) | `results/raw/swevo_e1_pymoo_corrected.csv` |
| E2 width / sparsity | **Done** (2160 runs) | `results/raw/e2_sparsity_full.csv` |
| E3 dimension (corrected) | **Done** (1920 runs) | `results/raw/swevo_e3_pymoo_corrected.csv` |
| E4 cross-regime meta | **Done** | `results/tables/e4_cross_regime_mean_ranks.csv`, `scripts/recompute_8method_suite.py` |
| Exact MILP / coverage | Done | `results/raw/exact_small.csv`, `results/target_coverage.csv` |

Legacy 10-method dumps (`e1_scaling_full.csv`, `e3_dimension_full.csv`, pre-sentinel E2 archive) remain for audit only. **Do not cite** unequal-budget DEBUG files such as `results/raw/e1_scaling.csv`.

### Headline numbers (six-solver core; optimization only)

| Axis | Mean FHR | Takeaway |
|------|----------|----------|
| E1 catalog size \(N\in\{5\mathrm{k},\ldots,210\mathrm{k}\}\) | **0.733–0.783** | Scale band is narrow under a 100k-eval ceiling |
| E2 width factor \(f\in\{0.5,1.0,1.5\}\) | **0.213 / 0.736 / 0.936** | Interval-box geometry spans the widest FHR range |
| E3 nutrient dim \(\|J\|\in\{6,9,12,16\}\) | **0.988 / 0.983 / 0.938 / 0.758** | High dimension separates solvers; dim 16 Random ≈ 0.32 while MA stays at 1 |

On the tight box (\(f=0.5\)): MA 0.458, NSGA-II 0.267, RandomRepair 0.242, Random 0.017 (RandomRepair–Random is descriptive; after Holm, MA vs Random is the main surviving contrast). Cross-regime order of the four shared solvers (MA, NSGA-II, SPEA2, Random) stays concordant (Kendall \(\tau=1\) on mean ranks). No universal champion is claimed.

## Quick start

```bash
pip install -r requirements.txt
python3 validate_data.py
python3 analyze_target_coverage.py
python3 run_exact.py configs/exact_small.yaml

# Corrected primary campaigns (process pool + resume)
python3 -u run_benchmark.py --config configs/swevo_e1_pymoo.yaml --stage main --workers 32
python3 -u run_e2.py --config configs/e2_full.yaml --stage main --workers 32
python3 -u run_e3.py --config configs/swevo_e3_pymoo.yaml --workers 32

# Recompute summaries / figures used in the empirical write-up
python3 scripts/recompute_8method_suite.py
```

Workers default to `min(32, ncpu-2)` or `NUTRIOPT_WORKERS` / yaml `workers:`.  
Checkpoints write `*.partial.csv`; restart resumes automatically.

## Experiments

| ID | Axis | Config | Runner | Primary rows |
|----|------|--------|--------|--------------|
| E1 | Food-space scale (nested NHANES pools) | `configs/swevo_e1_pymoo.yaml` | `run_benchmark.py` | 6 × 6 scales × 12 × 10 = **4320** |
| E2 | Interval width \(\{0.5,1.0,1.5\}\) | `configs/e2_full.yaml` | `run_e2.py` | 6 × 3 × 12 × 10 = **2160** |
| E3 | Nutrient dim \(\{6,9,12,16\}\) | `configs/swevo_e3_pymoo.yaml` | `run_e3.py` | 4 × 4 × 12 × 10 = **1920** |
| E4 | Cross-regime ranks / τ | analysis only | `scripts/recompute_8method_suite.py` | — |

Protocol: population 500, `max_evaluations=100000`, early-stop disabled, 12 representative targets × 10 hashed seeds. E1 uses `nested_sampling: true`. Result class for corrected E1/E3: `PRIMARY_CORRECTED_DECODER`.

## Algorithms

**Primary six:** MA, NSGA2, SPEA2, Greedy, RandomRepair, Random.  
**E3 four:** MA, NSGA2, SPEA2, Random.  
Reduced-pool MILP (CBC) calibrates attainability only. Older suite members (ACMO-Diet, BMOEA, RVEA, CTAEA) appear in legacy CSVs as literature/audit baselines, not in the corrected primary ranking. Mechanism tags: `docs/ALGORITHM_MECHANISM_MATRIX.md`.

## Documentation

| Path | Role |
|------|------|
| `docs/RCFA_FORMULATION.md` | Frozen problem regime |
| `docs/BENCHMARK_PROTOCOL.md` | Fair comparison |
| `docs/FAIRNESS_AUDIT.md` | Budget asymmetries (disclose) |
| `docs/CLAIM_EVIDENCE_MATRIX.md` | Claim ↔ evidence |
| `docs/METRIC_DEFINITIONS.md` | Metric definitions |
| `docs/BIBLIOGRAPHY_CLUSTER_MAP.md` | Survey bibliography clusters |
| `REPRODUCIBILITY.md` | Commands + configs |
| `literature/` | Survey protocol + seed matrix |

## Layout

```
configs/                 production YAMLs (incl. swevo_e*_pymoo.yaml)
run_benchmark.py         E1 fair runner (+ --workers, resume)
run_e2.py / run_e3.py
parallel_jobs.py         process-pool helper
simple_baselines.py / baseline_algorithms.py
sparsity_ops.py          E2 sentinel sanitization + width scaling
scripts/recompute_8method_suite.py
results/raw|tables|figures|statistics|logs
docs/                    protocols + claim matrices
```

## Claims discipline

- Optimization validity ≠ clinical dietary adequacy.
- No overall “best algorithm” / universal champion.
- Prefer corrected primary artifacts (`PRIMARY_CORRECTED_DECODER` / current E2 six-method CSV) over legacy 10-method dumps.
- Separate **algorithm search failure** from **intrinsic instance infeasibility**.
- Recipe ceiling regimes must not support ranking claims.
