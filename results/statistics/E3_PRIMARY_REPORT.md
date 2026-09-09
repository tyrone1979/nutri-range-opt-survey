# E3 PRIMARY Statistical Report

**Source:** `results/raw/e3_dimension_full.csv` (3840 rows, all `status=ok`)  
**Protocol:** NHANES-10K; dims \{6,9,12,16\}; 12 targets × 10 seeds × 8 algorithms; pop=500; max_evaluations=100000  
**Wall time:** ≈3.40 h (`12229.5s`) with 24 workers  
**Date:** 2026-09-08

## 1. Mean FHR by dimension × algorithm

| dim | ACMO-Diet | BMOEA | MA | NSGA2 | SPEA2 | RVEA | CTAEA | Random | **mean** |
|----:|----------:|------:|---:|------:|------:|-----:|------:|-------:|---------:|
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | 0.950 | **0.992** |
| 9 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 | 0.992 | 0.925 | 0.958 | **0.983** |
| 12 | 1.000 | 0.992 | 0.992 | 1.000 | 0.967 | 0.983 | 0.850 | 0.783 | **0.946** |
| 16 | 0.975 | 0.983 | **1.000** | 0.958 | 0.817 | 0.733 | 0.592 | **0.300** | **0.795** |

**Main effect:** mean FHR falls as nutrient dimension grows (especially 12→16).  
Machine table: `results/tables/e3_full_fhr_wide.csv`  
Figure: `results/figures/fig_e3_full_fhr_dim.png`

## 2. Friedman tests (target-aggregated FHR)

| dim | Friedman | p-value | Note |
|----:|---------:|--------:|------|
| 6 | 12.00 | 0.101 | n.s. (near-ceiling, little separation) |
| 9 | 26.82 | **3.6×10⁻⁴** | significant |
| 12 | 24.55 | **9.1×10⁻⁴** | significant |
| 16 | 50.72 | **1.0×10⁻⁸** | strongest separation |

## 3. Mean ranks at dim=16 (1 = best)

MA (2.96) ≈ ACMO-Diet (3.25) ≈ BMOEA (3.29) ≈ NSGA2 (3.50) ≫ SPEA2 (4.33) ≫ RVEA (5.33) ≫ CTAEA (6.38) ≫ Random (6.96)

## 4. Rank stability across dimensions

Largest rank range: **Random (2.25)**, CTAEA/MA (1.38).  
Most stable top-tier: SPEA2 range 0.25 (but mid-pack at high dim); NSGA2 0.83.  
Table: `results/tables/e3_full_rank_stability.csv`

## 5. Time-to-feasibility (median FFG_evals)

At dim=16, medians explode for weak methods (Random 100k = often never finds feasible; CTAEA 67k; SPEA2 26.5k) while ACMO/BMOEA/MA stay ~7–10k.

## 6. Budget fairness

Median `n_evals` = 100000 for all except MA (~100825, slight overshoot from local search). No RandomRepair/Greedy in this E3 config.

## 7. Reading (TEVC)

- Low-dim RCFA on NHANES-10K is near-saturated for strong EAs → rankings uninformative at dim=6.  
- **Constraint dimensionality bites at 16:** Random collapses; C-TAEA and RVEA degrade more than feasibility-biased/memetic methods.  
- Supports regime-dependent mechanism story: high-dim interval boxes stress feasibility discovery, not just Pareto refinement.  
- Note: missing nutrients were filled with `[0, empirical_cap]` for dim>original target support — disclose in manuscript.

## 8. Artifacts

| Type | Path |
|------|------|
| Raw | `results/raw/e3_dimension_full.csv` |
| Summary | `results/tables/e3_summary_by_dim_algo.csv` |
| FHR wide / ranks | `results/tables/e3_full_*.csv` |
| Figure | `results/figures/fig_e3_full_fhr_dim.{png,pdf}` |
