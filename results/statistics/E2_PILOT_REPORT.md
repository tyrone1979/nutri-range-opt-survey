# E2 Pilot Statistical Report (auto-generated companion)

**Source:** `results/raw/e2_pilot.csv`  
**Class:** PILOT (6 targets × 3 reps × 8 algorithms × 3 sparsity factors = 432 runs)  
**Budget:** pop=100, max_evaluations=4000, N=5000  
**Analysis:** `python3 analyze_e2.py --input results/raw/e2_pilot.csv --tag e2_pilot`

## 1. Mean FHR by sparsity × algorithm

| sparsity | ACMO-Diet | BMOEA | Greedy | MA | NSGA2 | Random | RandomRepair | SPEA2 |
|---------:|----------:|------:|-------:|---:|------:|-------:|-------------:|------:|
| 0.5 | 0.222 | 0.278 | 0.000 | 0.167 | 0.222 | 0.000 | 0.111 | 0.111 |
| 1.0 | 0.611 | 0.556 | 0.333 | 0.667 | 0.556 | 0.111 | 0.611 | 0.500 |
| 1.5 | 0.944 | 0.833 | 0.833 | 1.000 | 0.833 | 0.833 | 0.833 | 0.833 |

**Trend:** FHR increases as intervals loosen for essentially all methods.

## 2. Friedman tests (per sparsity; higher FHR better)

| sparsity | Friedman p | Interpretation (pilot n_targets=6 — underpowered) |
|---------:|-----------:|---------------------------------------------------|
| 0.5 | 0.170 | No significant overall difference |
| 1.0 | **0.0075** | Significant ranking differences |
| 1.5 | 0.429 | Methods converge (many near-ceiling FHR) |

## 3. Mean ranks (1 = best FHR)

At sparsity 1.0: ACMO-Diet ≈ BMOEA ≈ MA best; Random worst.  
Rank stability across sparsity: BMOEA most stable (rank range 0.5); Greedy most unstable (range 1.8).

Files: `results/tables/e2_pilot_mean_ranks.csv`, `e2_pilot_rank_stability.csv`

## 4. Budget fairness

| Algorithm | median n_evals | note |
|-----------|---------------:|------|
| BMOEA/ACMO/NSGA2/SPEA2/Random | 4000 | exact budget |
| MA | ~4130 | LS overshoot |
| Greedy | ~4085 | candidate scoring |
| RandomRepair | ~1999 | fills pop before hitting cap |

## 5. Figures

- `results/figures/fig_e2_pilot_fhr_sparsity.png`
- `results/figures/fig_e2_pilot_rank_sparsity.png`
- `results/statistics/e2_pilot_stats.json`

## 6. Caveat

PILOT only. Do **not** treat as TEVC primary ranking. Await `e2_sparsity_primary_lite.csv` (12×5×10×3, 20k evals).
