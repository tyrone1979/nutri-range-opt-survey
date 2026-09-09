# E2 PRIMARY (Full) Statistical Report

**Source:** `results/raw/e2_sparsity_full.csv` (3600 rows, all `status=ok`)  
**Protocol:** NHANES-10K; sparsity \{0.5, 1.0, 1.5\}; 12 targets × 10 seeds × 10 algorithms; pop=500; max_evaluations=100000  
**Wall time:** ≈1.38 h (`4963.6s`) with 32 workers  
**Finished:** 2026-09-08  
**Supersedes for ranking:** `e2_sparsity_primary_lite.csv` (20k×5) — keep LITE only as bridge/calibration.

## 1. Mean FHR

| \(f\) | ACMO | BMOEA | MA | NSGA2 | Repair | SPEA2 | RVEA | Greedy | CTAEA | Random | **mean** |
|------:|-----:|------:|---:|------:|-------:|------:|-----:|-------:|------:|-------:|---------:|
| 0.5 | 0.217 | 0.192 | **0.275** | 0.217 | 0.242 | 0.183 | 0.158 | 0.108 | 0.108 | **0.017** | **0.172** |
| 1.0 | 0.975 | 0.933 | **0.992** | 0.967 | 0.750 | 0.850 | 0.767 | 0.550 | 0.517 | 0.308 | **0.761** |
| 1.5 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.983 | 0.950 | 0.917 | 0.925 | 0.850 | **0.962** |

**Main effect:** FHR rises sharply as intervals loosen (0.17 → 0.76 → 0.96).  
Figure: `results/figures/fig_e2_full_fhr_sparsity.png`

## 2. vs PRIMARY_LITE (ΔFHR = Full − Lite)

Larger budget (100k vs 20k) and fixed Repair generally **raise** FHR, especially at \(f=1.0\) (e.g. Repair +0.35, NSGA2 +0.25, RVEA +0.27).  
Tight \(f=0.5\) gains are smaller (still hard). Repair at 0.5: +0.14 after equal-budget fix.

## 3. Friedman (target-aggregated FHR)

| \(f\) | Friedman | p |
|------:|---------:|--:|
| 0.5 | 26.33 | **1.8×10⁻³** |
| 1.0 | 60.36 | **1.1×10⁻⁹** |
| 1.5 | 22.82 | **6.6×10⁻³** |

All three significant; strongest separation at nominal sparsity \(f=1.0\).

## 4. Mean ranks (1 = best)

| \(f\) | Best | Worst |
|------:|------|-------|
| 0.5 | MA (4.21) | Random (6.75) |
| 1.0 | MA (3.54), ACMO (3.58), NSGA2 (3.67) | Random (8.58), CTAEA (7.67) |
| 1.5 | ACMO/BMOEA/NSGA2/MA/Repair (tie ~4.96) | Random (6.50) |

## 5. Rank stability across sparsity

Largest range: **Random (2.08)**, MA (1.42).  
Most stable mid: **RVEA (0.08)**, SPEA2 (0.54), Repair (0.62).  
Table: `results/tables/e2_full_rank_stability.csv`  
Figure: `results/figures/fig_e2_full_rank_sparsity.png`

## 6. Budget fairness

| Algorithm | median `n_evals` |
|-----------|-----------------:|
| Most EAs / Random / **RandomRepair** | **100000** |
| Greedy | ~100191 |
| MA | ~100808 |

Repair under-budget bug from LITE is **resolved** on full PRIMARY.

## 7. Reading (TEVC)

- Sparsity remains a first-class difficulty axis under full protocol.  
- At tight boxes, even 100k evals yield low FHR (~0.17 mean) — feasibility discovery dominates.  
- MA / ACMO / NSGA2 / BMOEA lead; Random last; CTAEA/Greedy mid-weak at \(f=1.0\).  
- Equal-budget Repair is competitive at \(f=0.5\) (rank ~5) and perfect at \(f=1.5\).

## 8. Artifacts

| Type | Path |
|------|------|
| Raw | `results/raw/e2_sparsity_full.csv` |
| Tables | `results/tables/e2_full_*.csv`, `e2_summary_by_sparsity_algo.csv` |
| Figures | `results/figures/fig_e2_full_fhr_sparsity.*`, `fig_e2_full_rank_sparsity.*` |
| This report | `results/statistics/E2_PRIMARY_REPORT.md` |
