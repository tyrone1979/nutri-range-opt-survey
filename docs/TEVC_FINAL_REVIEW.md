# TEVC Final Simulated Review

**Manuscript:** `manuscript/nutriopt_bench.tex` (post E1–E4 PRIMARY reconstruction)  
**Date:** 2026-09-09

## Reviewer 1 — Evolutionary Computation

| Question | Assessment | Action |
|----------|------------|--------|
| Genuine EC research question? | **Yes** — difficulty sensitivity vs ranking sensitivity | Keep as hook |
| Benchmark more than leaderboard? | **Yes** — controlled axes + concordance analysis | Emphasize E4 |
| Difficulty-vs-ranking clear? | **Yes** after rewrite; conceptual Fig.~1 | Done |
| Mechanism interpretation supported? | **Partial** — ranks consistent with feasibility bias; no ablations | Marked interpretive; CRITICAL if causal language slips |
| Trajectory evidence? | **Limited** — only probe CSVs | Explicitly limited in §7.6 |

**CRITICAL fixed:** Removed rank-reversal / universal-champion narratives.  
**MAJOR remaining:** Full generation trajectories not in PRIMARY logs; ablations absent.

## Reviewer 2 — Optimization

| Question | Assessment | Action |
|----------|------------|--------|
| RCFA well-defined? | **Yes** — frozen formulation | Cite RCFA doc |
| Feasibility / infeasibility handled? | **Yes** — MILP calibration + FHR | Retain §7.7 |
| Budgets fair? | **Mostly** — residuals disclosed | Keep disclosure |
| Exact calibration adequate? | **Yes as calibration, not competitor** | Correct framing |

**MAJOR:** CTAEA/pymoo residual asymmetries remain in Limitations.  
**MAJOR:** E3 empirical-cap padding disclosed.

## Reviewer 3 — Application / Benchmark

| Question | Assessment | Action |
|----------|------------|--------|
| Targets representative? | **Proxy only** — cluster vs random NN | Soft claim |
| Datasets separated? | **Yes** — recipe vs NHANES | Enforced |
| Recipe interpretation? | **Ceiling / no ranking** | Explicit |
| Clinical claims avoided? | **Yes** | Abstract/Conclusion clean |

## Fixes applied without inventing evidence

1. Central narrative → difficulty vs ranking (not reversal).  
2. E2 before E1 in Results (strongest axis first).  
3. Negative E1 scale result stated carefully.  
4. E3 “threshold-like practical effect,” not curse-of-dimensionality law.  
5. E4 Kendall as primary cross-regime claim; \(N^*\)/ADI demoted.  
6. Stats re-verified from raw PRIMARY CSVs.  
7. Conceptual figure added.

## Unfixable without new experiments

- Causal mechanism ablations for MA/ACMO.  
- Full PRIMARY search trajectories.  
- Completed PRISMA screening.  
- Perfect operator symmetry across all algorithms.
