# NutriOpt-Bench PRIMARY Report: E1–E4

**Date:** 2026-09-09  
**Scope:** Full fair-budget PRIMARY campaigns + cross-regime meta-analysis (E4)  
**Problem:** Range-Constrained Food Assembly (RCFA) — binary subset selection under nutrient interval boxes (+ optional meals)  
**Claim rule:** Optimization metrics only; no clinical validity. Recipe and NHANES analyzed separately.

---

## 0. Executive summary

| Experiment | Axis | Status | Rows | Wall clock |
|------------|------|--------|-----:|-----------:|
| **E1** | Food-space scale | **PRIMARY complete** | 8400 | ≈6.7 h (24 workers) |
| **E2** | Interval sparsity \{0.5,1.0,1.5\} | **PRIMARY complete** | 3600 | ≈1.4 h (32 workers) |
| **E3** | Nutrient dim \{6,9,12,16\} | **PRIMARY complete** | 3840 | ≈3.4 h (24 workers) |
| **E4** | Cross-regime alignment | **META complete** | — | analysis only |

**Headline findings**

1. **Sparsity is the sharpest difficulty axis.** E2 mean FHR: 0.17 → 0.76 → 0.96 as \(f: 0.5\to1.0\to1.5\).  
2. **Scale (NHANES) does not collapse FHR** under 100k evals (≈0.76–0.82 across 5K–full); **algorithm gaps dominate**.  
3. **Dimension hurts mainly at 16 nutrients** (mean FHR 0.99→0.80); low-dim is near ceiling.  
4. **Stable leaders across regimes:** MA, ACMO-Diet, NSGA2, BMOEA. **Stable laggards:** Random; CTAEA/Greedy weak on E1/E2.  
5. **E4:** Pairwise Kendall \(\tau\) on mean ranks among 8 shared algorithms is **high and significant** (E1–E2 \(\tau\approx0.98\); E1–E3 \(\tau\approx0.93\); E2–E3 \(\tau\approx0.91\)) — rankings are **largely aligned**, not reversed, across axes.  
6. Recipe catalogs (E1) hit **FHR=1.0 for all algorithms** — easy regime only; **do not use for ranking**.

---

## 1. Shared protocol

| Item | PRIMARY value |
|------|----------------|
| Population | 500 |
| `max_evaluations` | 100000 |
| Early stop on feasible count | **disabled** |
| Targets | 12 representatives (recipe side: 6) |
| Seeds / repeats | 10 |
| Primary metric | **FHR** (then FFG_evals, FSC; BestFit secondary) |
| Runners | `run_benchmark.py` / `run_e2.py` / `run_e3.py` + `parallel_jobs.py` |

**Fairness residuals (disclose):** MA local search and Greedy candidate scoring slightly overshoot median `n_evals` (~100.8k / ~100.2k). RandomRepair uses full 100k on E1/E2 full (LITE under-budget bug fixed).

**Raw PRIMARY files**

- `results/raw/e1_scaling_full.csv`  
- `results/raw/e2_sparsity_full.csv`  
- `results/raw/e3_dimension_full.csv`  

Per-experiment detail reports: `results/statistics/E{1,2,3}_PRIMARY_REPORT.md`.

---

## 2. E1 — Scaling

### 2.1 Design
8 scales: Foods.com (~1.6K), Allrecipes (~1.5K), NHANES \{5K,10K,25K,50K,100K,full≈210K\} × 10 algorithms × 10 seeds.

### 2.2 Recipe catalogs
Mean FHR = **1.000** for every algorithm on both catalogs.  
→ Ceiling / easy 4-nutrient regime. **Excluded from ranking and from E4 NHANES-scaled meta.**

### 2.3 NHANES mean FHR (primary)

| N | ACMO | BMOEA | MA | NSGA2 | SPEA2 | Repair | RVEA | Greedy | CTAEA | Random | mean |
|--:|-----:|------:|---:|------:|------:|-------:|-----:|-------:|------:|-------:|-----:|
| 5K | 0.983 | 0.958 | 0.967 | **0.992** | 0.917 | 0.817 | 0.750 | 0.575 | 0.567 | 0.325 | 0.785 |
| 10K | 0.950 | 0.975 | **1.000** | 0.933 | 0.858 | 0.742 | 0.733 | 0.508 | 0.542 | 0.325 | 0.757 |
| 25K | **0.992** | 0.950 | 0.975 | 0.975 | 0.892 | 0.925 | 0.842 | 0.658 | 0.608 | 0.417 | 0.823 |
| 50K | **0.975** | 0.933 | 0.967 | 0.883 | 0.858 | 0.792 | 0.775 | 0.600 | 0.558 | 0.342 | 0.768 |
| 100K | 0.950 | 0.958 | **0.983** | 0.942 | 0.850 | 0.800 | 0.825 | 0.600 | 0.617 | 0.325 | 0.785 |
| full | 0.975 | **0.983** | **0.983** | 0.950 | 0.833 | 0.858 | 0.867 | 0.608 | 0.558 | 0.333 | 0.795 |

Friedman significant at **all** NHANES scales (\(p\sim10^{-9}\)–\(10^{-10}\)).  
Leaders: MA / ACMO / NSGA2 / BMOEA. Worst: Random (mean rank ≈8.5, range 0.12).

**Figure:** `results/figures/fig_e1_full_fhr_scale.png`

---

## 3. E2 — Sparsity (centerpiece)

### 3.1 Design
NHANES-10K; sparsity factors \{0.5, 1.0, 1.5\} with sanitized one-sided bounds; 10 algorithms × 12 × 10.

### 3.2 Mean FHR

| \(f\) | ACMO | BMOEA | MA | NSGA2 | Repair | SPEA2 | RVEA | Greedy | CTAEA | Random | **mean** |
|------:|-----:|------:|---:|------:|-------:|------:|-----:|-------:|------:|-------:|---------:|
| 0.5 | 0.217 | 0.192 | **0.275** | 0.217 | 0.242 | 0.183 | 0.158 | 0.108 | 0.108 | **0.017** | **0.172** |
| 1.0 | 0.975 | 0.933 | **0.992** | 0.967 | 0.750 | 0.850 | 0.767 | 0.550 | 0.517 | 0.308 | **0.761** |
| 1.5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | 0.950 | 0.917 | 0.925 | 0.850 | **0.962** |

Friedman: all three factors significant (strongest at \(f=1.0\), \(p\sim10^{-9}\)).  
vs PRIMARY_LITE: full budget raises FHR (Repair at \(f=1.0\) ≈ +0.35 after equal-budget fix).

**Figures:** `fig_e2_full_fhr_sparsity.png`, `fig_e2_full_rank_sparsity.png`

---

## 4. E3 — Constraint dimensionality

### 4.1 Design
NHANES-10K; dims \{6,9,12,16\} via canonical nutrient prefix; missing target nutrients filled with \([0,\mathrm{empirical\_cap}]\) (disclose).  
8 algorithms (no Greedy / RandomRepair in `e3_full.yaml`).

### 4.2 Mean FHR

| dim | ACMO | BMOEA | MA | NSGA2 | SPEA2 | RVEA | CTAEA | Random | **mean** |
|----:|-----:|------:|---:|------:|------:|-----:|------:|-------:|---------:|
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | 0.950 | **0.992** |
| 9 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 | 0.992 | 0.925 | 0.958 | **0.983** |
| 12 | 1.000 | 0.992 | 0.992 | 1.000 | 0.967 | 0.983 | 0.850 | 0.783 | **0.946** |
| 16 | 0.975 | 0.983 | **1.000** | 0.958 | 0.817 | 0.733 | 0.592 | **0.300** | **0.795** |

Friedman n.s. at dim=6 (ceiling); significant at 9/12/16 (strongest at 16).  
At dim=16: MA/ACMO/BMOEA/NSGA2 ≫ Random/CTAEA/RVEA.

**Figure:** `results/figures/fig_e3_full_fhr_dim.png`

---

## 5. E4 — Cross-regime meta-analysis

**Type:** No new solver runs. Inputs = E1 NHANES + E2 full + E3 full.  
**Scripts:** `e4_alignment.py` → `results/e4_analysis/`; additional rank alignment → `results/tables/e4_*.csv`.

### 5.1 Regime difficulty (mean FHR over all algorithms)

Hardest → easiest cells:

| Rank | Cell | Mean FHR |
|-----:|------|--------:|
| 1 | **E2 \(f=0.5\)** | **0.172** |
| 2–8 | E1 scales / E3 \(d=16\) | 0.76–0.82 |
| … | E3 \(d=12\), E2 \(f=1.5\), E3 \(d\leq9\) | 0.95–0.99 |

→ **Tight intervals** are the distinctive stress test; large \(N\) alone is not.

### 5.2 Cross-regime mean Friedman ranks (1 = best)

| Algorithm | E1 NHANES | E2 sparsity | E3 dim | **Overall** |
|-----------|----------:|------------:|-------:|------------:|
| **MA** | 3.69 | 4.24 | 3.89 | **3.94** |
| **ACMO-Diet** | 3.81 | 4.50 | 3.90 | **4.07** |
| **NSGA2** | 3.96 | 4.50 | 3.96 | **4.14** |
| **BMOEA** | 4.24 | 4.74 | 4.04 | **4.34** |
| SPEA2 | 4.78 | 5.22 | 4.23 | 4.74 |
| RandomRepair | 4.90 | 5.21 | — | 5.06 |
| RVEA | 5.94 | 6.08 | 4.59 | 5.54 |
| CTAEA | 7.88 | 6.82 | 5.79 | 6.83 |
| Greedy | 7.28 | 6.42 | — | 6.85 |
| **Random** | 8.52 | 7.28 | 5.60 | **7.13** |

**Figure:** `results/figures/fig_e4_cross_regime_ranks.png`  
**Table:** `results/tables/e4_cross_regime_mean_ranks.csv`

### 5.3 Ranking concordance (Kendall \(\tau\), 8 shared algorithms)

| Pair | \(\tau\) | \(p\) |
|------|--------:|------:|
| E1 vs E2 | **0.982** | 7.6×10⁻⁴ |
| E1 vs E3 | **0.929** | 4.0×10⁻⁴ |
| E2 vs E3 | **0.909** | 1.8×10⁻³ |

Interpretation: **no wholesale rank reversal** across scale / sparsity / dimension for the shared set. Mechanism family ordering is robust; absolute FHR levels still regime-dependent.

### 5.4 Legacy E4 diagnostics (`e4_alignment.py`)

| Metric | Result | Note |
|--------|--------|------|
| Curve-collapse \(\sigma_\mathrm{align}\) (NHANES FHR_norm vs log N) | ≈0.047 | “Moderate consistency” — algorithms do **not** collapse to one curve |
| ADI (lower = more cross-exp consistent) | Greedy/MA/SPEA2/NSGA2 ~0.13–0.14; Random highest | Relative-level consistency ≠ ranking quality |
| Logistic “phase transition” \(N^*\) fits | Poor / unstable for top EAs | FHR flat in \(N\) → **do not claim a scaling phase boundary** from these fits |

**Figures:** `fig_e4_1_curve_collapse.png`, `fig_e4_2_adi_bar.png`, `fig_e4_3_phase_transition.png`

---

## 6. Mechanism reading (TEVC)

| Family | Evidence across E1–E3 |
|--------|------------------------|
| Memetic / feasibility-biased (MA, ACMO, BMOEA) | Consistently top ranks; strong under tight sparsity and high dim |
| Dominance MOEA (NSGA2) | Near-top across regimes |
| Indicator (SPEA2) | Mid; larger rank movement on some axes |
| Reference / constrained (RVEA, CTAEA) | Mid–weak; CTAEA weak on E1/E2 FHR |
| Construction / repair baselines | Repair competitive when budget-fair; Greedy weak |
| Random | Stably worst — validates combinatorial difficulty |

**Practical selection (optimization only):** prefer MA/ACMO/BMOEA/NSGA2 when feasible-region search is hard (\(f=0.5\) or high dim); do not trust recipe-scale ceilings; use MILP on reduced pools to detect intrinsic infeasibility before blaming heuristics.

---

## 7. Limitations

1. E3 nutrient padding with empirical caps for missing dimensions — document in manuscript.  
2. CTAEA wiring / pymoo vs native scalar objectives remain residual asymmetries (`docs/FAIRNESS_AUDIT.md`).  
3. E4 logistic \(N^*\) model is **not** supported by flat NHANES FHR — omit as a primary claim.  
4. No preference/cost/allergy axes in PRIMARY; diversity is soft-only.  
5. Literature seed matrix ≠ completed PRISMA survey.  
6. Exact MILP (12 targets, cand_limit=300): **4/12 intrinsically infeasible** — separate instance emptiness from algorithm failure.

---

## 8. Artifact index

| Item | Path |
|------|------|
| **This report** | `results/statistics/E1_E4_PRIMARY_REPORT.md` |
| E1 / E2 / E3 detail | `results/statistics/E1_PRIMARY_REPORT.md`, `E2_PRIMARY_REPORT.md`, `E3_PRIMARY_REPORT.md` |
| E4 analysis dir | `results/e4_analysis/` |
| Cross-regime ranks | `results/tables/e4_cross_regime_mean_ranks.csv` |
| Difficulty table | `results/tables/e4_regime_difficulty.csv` |
| Key figures | `results/figures/fig_e{1,2,3}_full_*.png`, `fig_e4_*.png` |
| Interim (pre-PRIMARY) | `results/statistics/INTERIM_REPORT_2026-09-08.md` (historical) |

---

## 9. Recommended manuscript claims (supported)

| Claim | Support |
|-------|---------|
| Interval sparsity strongly modulates FHR | E2 PRIMARY |
| Algorithm rankings differ significantly within regimes | Friedman E1/E2/E3 |
| Feasibility-biased/memetic methods + NSGA2 form a stable leading group | Cross-regime ranks + Kendall |
| NHANES scale alone is not a sharp FHR phase transition at 100k evals | E1 NHANES table |
| High nutrient dimension degrades weak methods | E3 dim=16 |
| Recipe easy-regime ≠ NHANES behavior | E1 split analysis |

| Claim | Status |
|-------|--------|
| Single overall champion | **Forbidden** |
| Clinical dietary adequacy | **Forbidden** |
| Universal FHR–scale collapse / clean \(N^*\) boundary | **Not supported** |
| Legacy `e1_scaling.csv` as evidence | **Forbidden (DEBUG)** |
