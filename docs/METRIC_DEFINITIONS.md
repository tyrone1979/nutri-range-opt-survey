# METRIC_DEFINITIONS.md

All definitions match the production evaluator in `common.evaluate_batch`,
with geometry/dynamics from `feasibility_geometry.py`.

## FHR — Feasible Hit Rate

Fraction of independent repeats (seeds) that return **at least one feasible** individual in the **final evaluated population** (or archive, if an algorithm maintains one — native EAs use final population).

\[
\mathrm{FHR} = \frac{1}{R}\sum_{r=1}^{R} \mathbf{1}\{\exists\, \mathcal{S}\ \mathrm{feasible\ in\ run}\ r\}
\]

Legacy note: older code set `fhr` from a single run as 0/1; aggregate FHR across repeats in analysis.

## FFG — First Feasible Generation

1-based generation index when the first feasible individual is observed. If none: `n_generations+1` (censored).

## FFG_evals — First Feasible Evaluations (preferred)

Number of objective evaluations consumed when the first feasible individual is found. If none: `max_evaluations` (censored). Prefer this for fair cross-algorithm comparison.

## FSC — Feasible Solution Count

Number of **feasible individuals in the final population** (not cumulative unique feasibles over the run, unless an algorithm explicitly archives them — then document archive size).

## BestFit

Maximum scalar fitness \(f(\mathcal{S})\) among evaluated final-population members (native path). For pymoo, map from \(f_1,f_2\) to the same scalar fitness via `evaluate_batch` on decoded individuals.

## Diversity / Contrib

Mean category diversity and contribution diversity over **feasible** final individuals; 0 if none feasible.

## Runtime

Wall-clock seconds for the solver call (single run).

## n_evals

Counted objective/batch evaluations attributed to the run. Report **actual** `n_evals` (may slightly exceed nominal budget for MA/Greedy inner scoring).

## Constraint violation (CV_nut / \(V_{\mathrm{nut}}\))

Nutrient interval violation of a solution (see `docs/RCFA_FORMULATION.md`). Dynamics logs use best \(V_{\mathrm{nut}}\) in the current population.

## Feasible-region density \(\rho\) (Monte Carlo)

Under the **same random subset sampling distribution** used by Random initialization (not the uniform measure on \(\{0,1\}^N\)):

\[
\hat\rho = \frac{1}{S}\sum_{s=1}^{S}\mathbf{1}\{\text{sample }s\text{ is feasible}\}.
\]

Also report mean/median/\(P_{90}\) of \(V_{\mathrm{nut}}\) on the same samples (`estimate_feasibility.py`).  
**Interpretation:** \(\hat\rho\) is a sampler-dependent difficulty proxy for E2, not a true volume ratio.

## Distance-to-feasibility proxy

For infeasible samples/individuals: \(V_{\mathrm{nut}}\) (and meal deficit when enabled). Lower is closer to nutrient feasibility.

## Trajectory metrics (dynamics)

Logged vs `n_evals` / generation:

- `frac_feasible` — fraction feasible in current population  
- `best_violation` — min \(V_{\mathrm{nut}}\) in population  
- `best_fitness` — max scalar fitness  
- `mean_diversity` — mean category diversity among feasibles (0 if none)  
- `n_feasible` — count feasible  

## Reporting rules

- Never claim superiority on one metric alone.  
- Report median + IQR and/or mean±SD with sample size.  
- Mark censored FFG values in survival-style summaries when used.  
- Always pair algorithm metrics with \(\hat\rho\) (or sparsity factor) in E2 analyses.
