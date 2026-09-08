# E2 Sparsity Protocol (STEP 3)

## Problem fixed

Naive center-scaling on sentinel uppers (`99999`, `9999999`) produced nonsensical intervals and inflated `mean_violation` (~10^6).

## Production rule

1. Detect \(U \ge 10^5\) as effectively unbounded.  
2. Replace with empirical meal-total cap (P99 of random cardinality-feasible meals on the same food space).  
3. Then scale half-width about center by sparsity factor \(f\in\{0.5,1.0,1.5\}\).  
4. Estimate Monte Carlo feasible density \(\hat\rho\) on the **scaled sanitized** intervals.  
5. Run all algorithms with identical `max_evaluations` via `run_e2.py`.

## Commands

```bash
python3 estimate_feasibility.py --config configs/e2_density.yaml
python3 run_e2.py --config configs/e2_pilot.yaml --stage pilot
python3 run_e2.py --config configs/e2_primary_lite.yaml --stage main
# Final TEVC campaign (expensive):
python3 run_e2.py --config configs/e2_full.yaml --stage main
```

## Result classes

| Class | Config | Meaning |
|-------|--------|---------|
| PILOT | `e2_pilot.yaml` | Engineering validation |
| PRIMARY_LITE | `e2_primary_lite.yaml` | Multi-seed sparsity study (20k evals, 5 reps) |
| PRIMARY | `e2_full.yaml` | Full TEVC protocol (100k, 10 reps) |
