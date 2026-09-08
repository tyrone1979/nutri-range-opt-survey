# CLAIM_EVIDENCE_MATRIX.md

| Claim | Evidence | Experiment | Statistical support | Figure/Table | Status |
|-------|----------|------------|---------------------|----------|--------|
| Diet optimization formulations are structurally fragmented (representation × target geometry × constraints) | Seed literature matrix + taxonomy | Survey coding | Qualitative | Tab. literature / taxonomy | **Supported at seed level**; full PRISMA counts pending live search |
| RCDA (index/binary food assembly under intervals) differs from continuous LP diet models | Formulation docs + Stigler/Smith vs implementation | Audit | Logical | Problem formulation | Supported |
| Existing E1 CSV (154 rows) is not publication-grade evidence | Audit: targets<12, reps=1, unfair budgets | `results/raw/e1_scaling.csv` | n/a | Audit | **DEBUG_RESULT — do not cite as finding** |
| Cluster-12 representatives cover target space better than random-12 (NN distance) | `results/target_coverage.csv` | Target analysis | Descriptive | Fig. target coverage | Supported (coverage proxy only) |
| Some representative targets are intrinsically infeasible on restricted candidate sets | MILP CBC on 500-food sample: 4/12 infeasible | `exact_small.csv` | Exact | Tab. exact | Supported for that candidate limit |
| Under equal tiny budgets, algorithms may all miss feasibility on hard targets | Profile: FHR=0 for all on rep_000 @400 evals | `e1_profile.csv` | n/a | Profile table | Provisional; budget too small for ranking |
| Feasibility is strongly target-dependent; EA/greedy > random on mini-pilot mean FHR | `e1_minipilot.csv` (4 tgt × 2 seeds, 2400 evals) | Mini-pilot | Descriptive only | Tab. pilot | **PILOT only — not primary ranking** |
| Algorithm X is best for diet optimization | — | — | — | — | **Forbidden until primary 12×10 runs complete** |
| Sanitized E2 interval scaling removes sentinel-upper artifact | meanV decreases with factor after fix | `estimate_feasibility.py` + `sparsity_ops.py` | Descriptive | Density tables | Supported |
| Under equal budget, mean FHR increases as intervals loosen (pilot) | `e2_pilot.csv`: FHR(0.5)<FHR(1.0)<FHR(1.5) for most algos | E2 pilot (6 tgt×3 rep×8 algo) | Descriptive trend | `fig_e2_pilot_fhr_sparsity` | **PILOT only** |
| Random density \(\hat\rho\approx0\) can coexist with EA FHR>0 | Pilot ρ≈0 while FHR up to 1.0 at f=1.5 | E2 pilot | Descriptive | Density + FHR tables | Supported as sampler note |
| Under equal budget, mean FHR increases as intervals loosen (PRIMARY_LITE) | 1800-run table: FHR(0.5)≪FHR(1.0)<FHR(1.5) | `e2_sparsity_primary_lite.csv` | Friedman per factor in `e2_primary_lite_stats.json` | `fig_e2_primary_lite_fhr_sparsity` | **Supported at PRIMARY_LITE** (not full 100k×10) |
| Feasibility-biased/memetic methods rank higher under tight sparsity than Random/RVEA | Mean ranks at f=0.5 | PRIMARY_LITE | Friedman/ranks | rank tables | Supported at PRIMARY_LITE; pending full PRIMARY |
| Algorithm × sparsity interaction (rank changes) | Rank stability table shows nonzero rank-range for several algos | PRIMARY_LITE | Descriptive ranks | `e2_primary_lite_rank_stability.csv` | Suggestive; needs formal interaction model + full PRIMARY |
| NutriSwap is state-of-the-art | Legacy `output/` reports | Different protocol | — | — | **Out of scope / do not claim** |

Update this matrix after each production experiment stage.
