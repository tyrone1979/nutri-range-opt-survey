# BENCHMARK_PROTOCOL.md — NutriOpt-Bench

## 1. Design principle

NutriOpt-Bench compares **optimization mechanisms** on range-constrained diet assembly. It does **not** evaluate clinical health outcomes.

## 2. Computational budget (fairness — mandatory)

| Parameter | Production value | Notes |
|-----------|------------------|-------|
| Primary budget | **max_evaluations = 100_000** | Count every call that scores an individual |
| Population size | 200 (pilot) / **500** (primary) / 2000 (stress; optional) | Must be identical across algorithms |
| Generations | Derived: stop when evals exhausted | Do not give extra gens to any algo |
| Early stop on `#feasible ≥ 20` | **Disabled** in production | Was enabled in legacy code — unfair |
| Wall-clock | Recorded, not used as primary stop | Cap optional for MILP |

All algorithms receive the **same** `max_evaluations` and `pop_size` from YAML.

## 3. Random seeds

For experiment `exp`, dataset scale `s`, target id `t`, algorithm `a`, repeat `r`, base seed `S`:

\[
\mathrm{seed} = \mathrm{hash}(S, \mathrm{exp}, s, t, a, r) \bmod 2^{31}
\]

Implementation: `benchmark_core.make_seed(...)`.  
pymoo and NumPy/Python RNGs must both be seeded with this value.

Repeats: **r = 0…9** for primary; **r = 0…4** for robustness (45 targets).

## 4. Targets

| Tier | Targets | Repeats | Use |
|------|---------|---------|-----|
| Profiling | 1 | 1 | Runtime estimate |
| Pilot | 3 | 3 | Correctness |
| Primary | **12** representative | **10** | Main claims |
| Robustness | **45** | **5** | Sensitivity |
| Coverage | **100** | 1–3 | Target-space check |

Representative targets from clustering/stratification (`data/representative_targets*.csv`), with coverage report in `results/target_analysis/`.

## 5. Experiments

- **E1 Scaling:** Foods.com / Allrecipes / NHANES samples — **analyze separately** by data-generating process.  
- **E2 Sparsity:** width factors \{0.5, 1.0, 1.5\} on NHANES representative targets.  
- **E3 Dimension:** \{6, 9, 12, 16\} nutrient dims (project matrix + target).  
- **E4:** meta-analysis only (no new solver runs beyond aggregating E1–E3).

## 6. Algorithms (primary)

BMOEA, ACMO-Diet, MA, NSGA2, RVEA, CTAEA, SPEA2, RandomSearch, GreedyConstruct, MILP-small (exact on small instances).

## 7. Metrics

See `docs/METRIC_DEFINITIONS.md`. Primary: **FHR**, **FFG_evals**, **FSC**, BestFit, Diversity, Contrib, runtime, n_evals, nutrient CV.

## 8. Statistics

- Per condition: 10 seeds.  
- Friedman + Nemenyi (or Wilcoxon + Holm) on paired target–seed aggregates.  
- Effect sizes + CIs.  
- FHR: binomial / bootstrap CIs.  
Artifacts: `results/statistical_tests/`.

## 9. Software / hardware (fill at run time)

Record in `results/reproducibility/env.json`: Python, numpy, scipy, pandas, pymoo, PuLP/CBC versions; CPU model; cores; OS.

## 10. Failure handling

Failed runs log to `results/logs/` with exception traceback; CSV row marked `status=error`. **Never drop silently.**

## 11. Anti-cheating rules

Identical budgets; no post-hoc target removal; no cherry-picked seeds; no selective success reporting; no “best of” without full distribution.
