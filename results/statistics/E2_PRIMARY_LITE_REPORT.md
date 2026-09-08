# E2 PRIMARY_LITE Statistical Report

**Source:** `results/raw/e2_sparsity_primary_lite.csv` (1800 rows, all analyzed)  
**Protocol:** 12 targets × 5 seeds × 10 algorithms × 3 sparsity × NHANES-10K  
**Budget:** pop=200, max_evaluations=20000  
**Wall time:** ≈3.09 h (`Total time 11126.5s`)  
**Auto pipeline:** `watch_e2_and_analyze.py` → `analyze_e2.py --tag e2_primary_lite`

## 1. Mean FHR (from `e2_primary_lite_fhr_wide.csv`)

| sparsity | ACMO-Diet | BMOEA | MA | NSGA2 | SPEA2 | RVEA | CTAEA | Greedy | RandomRepair | Random |
|---------:|----------:|------:|---:|------:|------:|-----:|------:|-------:|-------------:|-------:|
| **0.5** | 0.150 | 0.150 | *(see wide)* | 0.233* | 0.133 | 0.133 | 0.083 | 0.050 | 0.100 | 0.017 |
| **1.0** | 0.817 | 0.750 | 0.917 | 0.717 | 0.633 | 0.500 | 0.400 | 0.333 | 0.400 | 0.200 |
| **1.5** | 0.983 | 0.967 | 1.000 | 1.000 | 0.933 | 0.867 | 0.883 | 0.817 | 0.967 | 0.633 |

\*Use machine table as ground truth:

```text
results/tables/e2_primary_lite_fhr_wide.csv
```

**Main effect:** FHR increases sharply as sparsity factor goes 0.5 → 1.0 → 1.5 for every algorithm.

## 2. Friedman tests (target-aggregated FHR)

| sparsity | Friedman stat | p-value |
|---------:|--------------:|--------:|
| 0.5 | 18.20 | **0.033** |
| 1.0 | 67.12 | **5.6×10⁻¹¹** |
| 1.5 | 35.00 | **6.0×10⁻⁵** |

Source: `results/statistics/e2_primary_lite_stats.json`

## 3. Mean ranks (1 = best)

| sparsity | Best (rank≈1) | Worst |
|----------|---------------|-------|
| 0.5 | ACMO-Diet (1.58), BMOEA (2.50) | SPEA2 (9.17), Random (8.33) |
| 1.0 | ACMO-Diet (1.42), MA (2.83), BMOEA (3.25) | Random (8.92) |
| 1.5 | ACMO-Diet (1.17), BMOEA (2.42) | Random (8.83) |

## 4. Rank stability across sparsity (interaction proxy)

| algorithm | mean rank | rank range across sparsity |
|-----------|----------:|---------------------------:|
| ACMO-Diet | 1.39 | **0.42** (most stable top) |
| RVEA | 6.50 | 0.50 |
| Random | 8.69 | 0.58 |
| BMOEA | 2.72 | 0.83 |
| SPEA2 | 7.94 | **3.00** (largest shift) |
| CTAEA | 4.89 | 2.67 |

**TEVC-relevant reading:** top feasibility-biased methods stay top; some MOEAs (SPEA2/CTAEA) change rank more across sparsity → evidence of **algorithm × sparsity interaction**, still short of full 100k×10 PRIMARY.

## 5. Budget fairness

| algorithm | median n_evals |
|-----------|---------------:|
| ACMO/BMOEA/NSGA2/RVEA/CTAEA/SPEA2/Random | 20000 |
| MA / Greedy | ~20130–20290 (slight overshoot) |
| RandomRepair | **~3999 (under-budget — must fix)** |

## 6. Artifacts

| Type | Path |
|------|------|
| Raw | `results/raw/e2_sparsity_primary_lite.csv` |
| Stats JSON | `results/statistics/e2_primary_lite_stats.json` |
| FHR / ranks / stability tables | `results/tables/e2_primary_lite_*.csv` |
| Figures | `results/figures/fig_e2_primary_lite_fhr_sparsity.png`, `..._rank_sparsity.png` |

## 7. Next

1. Fix RandomRepair evaluation accounting.  
2. Optional: formal sparsity×algorithm interaction model.  
3. Launch E1 scaling + `e2_full.yaml` when ready.
