# FINAL_REVIEW.md — Triple Reviewer Pass

## Reviewer A — Evolutionary Computation

| Question | Assessment | Action |
|----------|------------|--------|
| Fair budget? | Production runner enforces shared max_evals; legacy E1 CSV unfair | Keep DEBUG flag; only cite production CSVs |
| Algorithms characterized? | Table in audit; pymoo vs native documented | OK |
| Statistical comparisons valid? | Primary 12×10 not finished | Do not report significance yet |
| Mechanistic contribution? | Hypothesis ready; evidence partial (MILP infeasibility, coverage) | Complete E1–E3 before strong mechanism claims |

**Verdict:** Framework strong; ranking claims premature.

## Reviewer B — Operations Research

| Question | Assessment | Action |
|----------|------------|--------|
| Formulation rigorous? | Matches implementation; binary MILP provided | OK |
| Exact methods? | CBC via PuLP on small candidates | Note candidate_limit approximation |
| Nontrivial? | Several reps infeasible on restricted sets; profile FHR=0 | Supported |
| Simple baselines? | Random/Greedy/Repair added | OK |

**Verdict:** Acceptable OR positioning if candidate-subset MILP limitations are stated.

## Reviewer C — Nutrition / Health Informatics

| Question | Assessment | Action |
|----------|------------|--------|
| Clinical claims avoided? | Manuscript explicitly separates optimization vs clinical validity | Keep |
| Units documented? | Still weak in data docs | Add unit table before submission |
| Targets valid? | Some may be intrinsically infeasible on subsets | Always run MILP/bounding screen |
| Dataset description? | NHANES duplicate IDs explained as gram instances | OK |

**Verdict:** Cautious tone appropriate; add nutrient unit appendix.

## Residual limitations moved to manuscript §Limitations
- Incomplete PRISMA live search
- Full primary experiments pending
- No clinical validation
- Recipe vs NHANES heterogeneity
