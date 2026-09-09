# TEVC Claim--Evidence Matrix (Final PRIMARY)

**Updated:** 2026-09-09 after complete E1–E4 PRIMARY.  
**Rule:** RAW PRIMARY CSVs supersede summary reports on conflict.

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 1 | Interval constraint tightness strongly modulates FHR | **Supported** | E2 PRIMARY: mean FHR 0.172→0.761→0.962 |
| 2 | NHANES scale alone does not produce FHR collapse in 5K–210K under 100k evals | **Supported** | E1 NHANES mean FHR ≈0.76–0.82; no monotonic collapse |
| 3 | Constraint dimensionality has pronounced effect at 16 nutrients | **Supported** | E3: ceiling through 9; heterogeneous at 16 (mean 0.795) |
| 4 | Algorithm rankings differ significantly within regimes | **Supported** | Friedman significant on all E1 NHANES scales; E2 all \(f\); E3 dims 9/12/16 (dim 6 n.s. ceiling) |
| 5 | Cross-regime rankings are highly concordant | **Supported** | Kendall τ 0.982 / 0.929 / 0.909 (8 shared algs) |
| 6 | Leading group: MA / ACMO-Diet / NSGA-II / BMOEA | **Supported (descriptive)** | Cross-regime mean ranks 3.94–4.34 |
| 7 | Some small instances are intrinsically infeasible | **Supported** | EXACT: 4/12 infeasible under CBC |
| 8 | Cluster-12 has better coverage proxy than random-12 | **Supported (descriptive)** | NN 3.77 vs 4.54 |
| 9 | Universal champion algorithm | **FORBIDDEN** | — |
| 10 | Clinical dietary adequacy / health effectiveness | **FORBIDDEN** | — |
| 11 | Scaling phase boundary \(N^*\) / logistic phase transition | **NOT SUPPORTED** | Unstable fits; move to supplement only |
| 12 | Wholesale algorithm-rank reversal across regimes | **NOT SUPPORTED / RETRACTED** | Contradicted by high Kendall τ |
| 13 | “Curse of dimensionality” as blanket law | **FORBIDDEN overclaim** | Use threshold-like practical effect at 16D |
| 14 | Causal mechanism of MA local search | **Interpretive only** | Needs ablations; mark as hypothesis-consistent |
| 15 | Completed PRISMA systematic review | **NOT CLAIMED** | Seed matrix / structured analysis only |
| 16 | Recipe FHR for algorithm ranking | **FORBIDDEN** | Ceiling FHR=1.0 all methods |

## Verified key numbers (from raw CSVs 2026-09-09)

| Quantity | Value |
|----------|------:|
| E2 mean FHR \(f=0.5/1.0/1.5\) | 0.172 / 0.761 / 0.962 |
| E1 NHANES mean FHR range | 0.757–0.823 |
| E3 mean FHR dim 6/9/12/16 | 0.992 / 0.983 / 0.946 / 0.795 |
| Kendall E1–E2 / E1–E3 / E2–E3 | 0.982 / 0.929 / 0.909 |
| MILP infeasible reps | 000, 005, 008, 011 |
