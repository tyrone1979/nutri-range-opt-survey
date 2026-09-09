# E1 PRIMARY Statistical Report

**Source:** `results/raw/e1_scaling_full.csv` (8400 rows, all `status=ok`)  
**Protocol:** pop=500; max_evaluations=100000; 10 algorithms; 10 seeds  
**Design:** 2 recipe catalogs (6 targets each) + 6 NHANES scales × 12 targets  
**Wall time:** ≈6.67 h (`24029.6s`) with 24 workers  
**Date:** finished 2026-09-08

**Analyze separately by data-generating process** (protocol requirement).

---

## 1. Recipe catalogs (Foods.com / Allrecipes)

| Dataset | N | Mean FHR (all algos) |
|---------|--:|---------------------:|
| foods_com | 1622 | **1.000** |
| allrecipes | 1500 | **1.000** |

Ceiling effect: 4-nutrient recipe instances are too easy under 100k evals.  
**Do not rank algorithms on recipe FHR.** Useful only as easy-regime calibration.

Table: `results/tables/e1_full_fhr_wide_recipe.csv`

---

## 2. NHANES scaling (primary scientific axis)

### Mean FHR by scale × algorithm

| N | ACMO | BMOEA | MA | NSGA2 | SPEA2 | Repair | RVEA | Greedy | CTAEA | Random | **mean** |
|--:|-----:|------:|---:|------:|------:|-------:|-----:|-------:|------:|-------:|---------:|
| 5K | 0.983 | 0.958 | 0.967 | **0.992** | 0.917 | 0.817 | 0.750 | 0.575 | 0.567 | 0.325 | **0.785** |
| 10K | 0.950 | 0.975 | **1.000** | 0.933 | 0.858 | 0.742 | 0.733 | 0.508 | 0.542 | 0.325 | **0.757** |
| 25K | **0.992** | 0.950 | 0.975 | 0.975 | 0.892 | 0.925 | 0.842 | 0.658 | 0.608 | 0.417 | **0.823** |
| 50K | **0.975** | 0.933 | 0.967 | 0.883 | 0.858 | 0.792 | 0.775 | 0.600 | 0.558 | 0.342 | **0.768** |
| 100K | 0.950 | 0.958 | **0.983** | 0.942 | 0.850 | 0.800 | 0.825 | 0.600 | 0.617 | 0.325 | **0.785** |
| full (210K) | 0.975 | **0.983** | **0.983** | 0.950 | 0.833 | 0.858 | 0.867 | 0.608 | 0.558 | 0.333 | **0.795** |

**Scale main effect (mean FHR):** no sharp collapse with \(N\); overall stays ~0.76–0.82. Difficulty is not a simple “larger \(N\) ⇒ much lower FHR” under this budget.  
**Algorithm gap is large and stable:** memetic/feasibility-biased and NSGA2 ≫ Random/CTAEA/Greedy.

Machine table: `results/tables/e1_full_fhr_wide_nhanes.csv`  
Figure: `results/figures/fig_e1_full_fhr_scale.png`

### Friedman (target-aggregated FHR), NHANES only

All six scales **significant** (p ~ 10⁻⁹–10⁻¹⁰).

| N | Top-3 mean ranks (1=best) | Worst |
|--:|---------------------------|-------|
| 5K | NSGA2, MA, ACMO | Random, CTAEA |
| 10K | MA, BMOEA, ACMO | Random, Greedy |
| 25K | ACMO, NSGA2, MA | CTAEA, Random |
| 50K | ACMO, MA, BMOEA | Random, CTAEA |
| 100K | MA, NSGA2, ACMO | Random, CTAEA |
| full | MA, ACMO, BMOEA | Random, CTAEA |

### Rank stability across NHANES scales

| Algorithm | mean rank | rank range |
|-----------|----------:|-----------:|
| Random | 8.52 | **0.12** (stably worst) |
| NSGA2 | 3.96 | 0.62 |
| ACMO-Diet | 3.81 | 0.75 |
| MA | **3.69** | 0.88 |
| BMOEA | 4.24 | 1.04 |
| SPEA2 | 4.78 | **1.29** (largest shift among mid/top) |
| CTAEA | 7.88 | 1.21 |
| Greedy | 7.28 | 0.75 |
| RVEA | 5.94 | 1.12 |
| RandomRepair | 4.90 | 1.12 |

---

## 3. Budget fairness

| Algorithm | median `n_evals` | Note |
|-----------|-----------------:|------|
| Most EAs / Random / Repair | 100000 | Full budget |
| Greedy | ~100178 | Slight overshoot; some early stops below 100k (min ~77k) |
| MA | ~100806 | Local-search overshoot |

RandomRepair **now uses full budget** (unlike E2 PRIMARY_LITE under-use).

---

## 4. Reading (TEVC)

1. **Recipe vs NHANES must stay separated** — recipe FHR=1 is not evidence of algorithm quality.  
2. On NHANES, **scale alone does not induce a strong FHR phase transition** under 100k evals; **mechanism gaps dominate**.  
3. Feasibility-biased / memetic (MA, ACMO, BMOEA) and NSGA2 form a stable leading group; Random is stably last; CTAEA/Greedy weak.  
4. Complements E3: E1 stresses catalog size; E3 stresses constraint dimension — different axes, both regime-dependent.

---

## 5. Artifacts

| Type | Path |
|------|------|
| Raw | `results/raw/e1_scaling_full.csv` |
| NHANES FHR / ranks | `results/tables/e1_full_*nhanes*.csv` |
| Recipe FHR | `results/tables/e1_full_fhr_wide_recipe.csv` |
| Full summary | `results/tables/e1_summary_by_scale_algo.csv` |
| Figure | `results/figures/fig_e1_full_fhr_scale.{png,pdf}` |
| This report | `results/statistics/E1_PRIMARY_REPORT.md` |
