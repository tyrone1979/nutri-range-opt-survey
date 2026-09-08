# Phase II STEP 3 Completion Note

**Date:** 2026-09-08  
**Focus:** E2 feasible-region sparsity — fix scaling, production runner, pilot evidence

## Delivered

| Item | Path |
|------|------|
| Sanitized sparsity ops | `sparsity_ops.py` |
| E2 protocol doc | `docs/E2_SPARSITY_PROTOCOL.md` |
| Production E2 runner | `run_e2.py` |
| Configs | `e2_pilot.yaml`, `e2_primary_lite.yaml`, `e2_full.yaml`, `e2_density.yaml` |
| Density (sanitized) | `results/target_analysis/feasible_density.csv` |
| Pilot results | `results/raw/e2_pilot.csv` (432 rows, all `ok`) |
| Summaries | `results/tables/e2_summary_by_sparsity_algo.csv` |
| Figures | `results/figures/fig_e2_pilot_fhr_sparsity.*` |
| Primary-lite | launched via `e2_primary_lite.yaml` (background) |

## Bug fix

Sentinel uppers (`99999` / `9999999`) no longer use naive center-scaling.  
Finite dims: center half-width scale.  
One-sided dims: empirical P99 cap + \(L' = L / f\).

**Check:** mean violation now decreases with looser factor (0.5→1.0→1.5): ~20.5k → ~7.0k → ~6.0k.

## Pilot findings (PILOT only — not TEVC primary claims)

Mean FHR rises sharply as intervals loosen:

| Factor | Example mean FHR (MA) | Random |
|--------|----------------------:|-------:|
| 0.5 | 0.17 | 0.00 |
| 1.0 | 0.67 | 0.11 |
| 1.5 | 1.00 | 0.83 |

- Monte Carlo \(\hat\rho\) remains ≈0 even when EAs succeed → **structured search ≢ random density**.  
- Random / Greedy collapse under tight factor more than feasibility-biased EAs.  
- Equal budget largely held (`n_evals` ~1.5k–4.2k; MA/Greedy slight overshoot).

## Still pending

- Finish `PRIMARY_LITE` / full `PRIMARY` E2  
- E1 scaling PRIMARY  
- Formal stats + algorithm×sparsity interaction  
- Manuscript rewrite  

## Commands

```bash
python3 estimate_feasibility.py --config configs/e2_density.yaml
python3 -u run_e2.py --config configs/e2_pilot.yaml --stage pilot
python3 -u run_e2.py --config configs/e2_primary_lite.yaml --stage main
python3 -u run_e2.py --config configs/e2_full.yaml --stage main
```
