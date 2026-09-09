# Revision Status Audit — budget+E5 closed

## Budget sensitivity (COMPLETE)
File: `results/raw/e_budget_f05.csv` (2880 rows; 25/50 from live runs; 100k from PRIMARY E2; 200k from `e_budget_200k.csv`)
Figure: `results/figures/fig_budget_vs_fhr.{png,pdf}`

| Budget | MA | ACMO | BMOEA | NSGA2 | RR | Random | mean(6) |
|--------|-----|------|-------|-------|-----|--------|---------|
| 25k | 0.192 | 0.183 | 0.175 | 0.133 | 0.142 | 0.008 | 0.139 |
| 50k | 0.200 | 0.200 | 0.208 | 0.183 | 0.192 | 0.008 | 0.165 |
| 100k | 0.275 | 0.217 | 0.192 | 0.217 | 0.242 | 0.017 | 0.193 |
| 200k | 0.267 | 0.242 | 0.242 | 0.242 | 0.250 | 0.008 | 0.208 |

Interpretation: FHR rises with budget for guided methods; 100k→200k lift is modest (~+0.015 overall). PRIMARY 100k is not an under-budget artifact. Random stays near floor.

## E5 synthetic (COMPLETE)
File: `results/raw/e5_synthetic.csv` (540 runs)
Mean FHR by f: 0.667 / 0.794 / 0.917; by m: 0.930 vs 0.656; by N: ~0.79–0.81 flat.

## Manuscript
Budget + E5 numbers written into Results/Discussion. PRIMARY 100k protocol unchanged.
