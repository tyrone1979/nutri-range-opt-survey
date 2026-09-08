# FAIRNESS_AUDIT.md — NutriOpt-Bench (TEVC)

**Date:** 2026-09-08  
**Scope:** Production path `run_benchmark.py` + `configs/*_full.yaml` + baselines/MILP  
**Non-scope for claims:** legacy `e1_scaling.py` + `results/raw/e1_scaling.csv` + `output/*` NutriSwap runs

---

## 1. Audit summary

| Area | Production path | Legacy path | Action |
|------|-----------------|-------------|--------|
| Max evaluations | Shared via `RunBudget.max_evaluations` | Unequal (early stop ≥20 feas) | Use production only |
| Population size | Shared YAML | Hardcoded / ignored YAML | Use production only |
| Early stop | `early_stop_feasible: null` | Enabled | Keep null for PRIMARY |
| Seeds | `make_seed(...)` + pymoo `seed=` | pymoo fixed 42 | Production OK |
| Target selection | Representative CSV | Ad-hoc “widest” filter in old loader | Prefer representatives |
| Hidden tuning | None intended | NutriSwap domain ops | Exclude NutriSwap from main |
| Preprocessing | Shared loaders | Shared | Document meal-map drop |

**Verdict:** PRIMARY experiments **must** use `run_benchmark.py`. Legacy CSVs are **not fair**.

---

## 2. Preprocessing (shared)

| Step | Behavior | Fairness risk | Mitigation |
|------|----------|---------------|------------|
| NHANES load | Drop rows without meal map; fill NaN nutrients with 0; drop cols with >50% missing | Changes \(N\) and nutrient set | Same for all algos; report final \(N\), \(m\) |
| Duplicate `food_id` | Kept as distinct gram instances | Inflates \(N\) vs unique foods | Disclose in manuscript |
| Recipe loaders | 4 nutrients; no meal constraint | Different regime vs NHANES | **Do not pool** into one scaling curve |
| Target align | Map nutrient names / synonyms | Recipe targets generated differently | Separate analysis blocks |

No algorithm-specific preprocessing.

---

## 3. Normalization

| Item | Status |
|------|--------|
| Feature normalization of \(A\) | **None** |
| Objective normalization (pymoo) | Algorithm-internal (e.g., RVEA) only |
| Target scaling in E2 | Explicit width factor applied equally |

Risk: nutrient scales differ (kcal vs mg). Acceptable if identical across algorithms; do not compare absolute \(V_{\mathrm{nut}}\) across differently scaled target sets without normalization notes.

---

## 4. Target construction

| Protocol | Fair? | Notes |
|----------|-------|-------|
| `representative_targets*.csv` clustering | Yes if fixed before runs | Freeze file + seed; no post-hoc removal of hard targets |
| E2 width scaling about interval center | Yes | Same factor for all algos |
| Small-dataset percentile targets | Yes within recipe block | Not interchangeable with NHANES targets |
| Dropping infeasible targets after seeing EA fails | **Forbidden** | Use MILP pre-check; label intrinsic infeasibility |

---

## 5. Initialization

| Algo | Init | Equal information? |
|------|------|--------------------|
| BMOEA / MA / Random / Repair | Uniform random cardinality | Yes |
| ACMO-Diet | 70% meal-aware | **Mechanism difference** (documented), not a hidden budget cheat |
| Greedy | Meal seeds + constructive | Mechanism difference |
| pymoo | `IntSampling` uniform | Yes across pymoo algos |

Initialization differences are **part of the algorithm definition**, not fairness violations — but must not be confused with unequal eval budgets.

---

## 6. Random seeds

Production:

```text
seed = make_seed(base, exp, scale, target_id, algorithm, rep)
np.random.seed(seed); random.seed(seed)
pymoo algorithm.setup(..., seed=seed)
```

Legacy issue (fixed in production): pymoo `seed=42` constant.

Requirement: ≥10 independent `rep` values for PRIMARY.

---

## 7. Evaluation & stopping

| Rule | Production |
|------|------------|
| Primary budget | Identical `max_evaluations` |
| Soft gen cap | `max_generations` high; evals dominate |
| Early stop on feasible count | **Disabled** (`null`) |
| Recorded | actual `n_evals`, wall-clock `runtime`, termination via budget |

### Known residual asymmetries (disclose, do not hide)

1. **MA local search** and **Greedy candidate scoring** may count inner evaluations → `n_evals` can slightly exceed nominal budget. Report actual `n_evals`; optional sensitivity: disable LS / limit greedy candidates.  
2. **pymoo** path re-decodes population each generation for FHR tracking → extra `evaluate_single` calls beyond pymoo’s internal eval accounting in some versions; production increments `n_evals += pop_size` per generation for budget parity with native loops.  
3. **Final** `evaluate_batch` after loop (native) adds one more pop eval in some algos — verify and standardize in instrumentation pass.  
4. **Fixed (RandomRepair):** previously stopped once `pop_size` repaired individuals were built → median `n_evals` ~4k vs 20k budget on PRIMARY_LITE. Now samples/repairs until `max_evaluations` (elite archive of size `pop_size`). **E2 PRIMARY_LITE Repair rows remain under-budget; re-run Repair (or full E2) before ranking.**  
5. **GreedyConstruct** still stops at `pop_size` constructions (same class of under-use); disclose until aligned.

---

## 8. Repair & operators

| Mechanism | Who has it | Fairness stance |
|-----------|------------|-----------------|
| RoundingRepair | All pymoo | Shared among pymoo |
| RandomRepair baseline | Explicit repair algo | Comparative mechanism |
| Cardinality clipping | ACMO / others | Algorithmic |
| No NutriSwap fingerprint ops in primary | — | Prevents domain-biased main table |

---

## 9. Parameters

PRIMARY defaults (`configs/e1_full.yaml`):

| Param | Value |
|-------|-------|
| `population_size` | 500 |
| `max_evaluations` | 100000 |
| `early_stop_feasible` | null |
| `targets_total` | 12 |
| `repeats` | 10 |
| `seed` | 42 |

No per-algorithm hyperparameter search is allowed for the main table.  
If a sensitivity study tunes one algorithm, report separately and do not mix into primary ranks.

---

## 10. Hardware / parallelism

| Item | Policy |
|------|--------|
| Hardware | Record in `results/reproducibility/env.json` |
| Parallel workers | If used, must not change per-run eval budget |
| Wall-clock | Secondary metric; not primary stop |

---

## 11. Selective reporting (anti-cheating checklist)

- [ ] No dropping difficult targets after EA results  
- [ ] No unequal `max_evaluations`  
- [ ] No cherry-picked seeds  
- [ ] No reporting only best of \(R\) runs without distribution  
- [ ] Failed runs logged with `status=error`  
- [ ] Intrinsically infeasible instances labeled via MILP/bounds  
- [ ] NutriSwap not in primary leaderboard  

---

## 12. Representation inequivalence (scientific, not budget cheat)

Native variable-length sets vs pymoo padded integers change variation operators.  
**Hard feasibility definition is shared.**  
Manuscript must discuss this as algorithm encoding difference, not as an error — unless one encoding receives extra evals (then fix).

---

## 13. CTAEA special note

CTAEA is a constrained MOEA, but `DietProblem` sets `n_constr=0`.  
This may under-activate constraint archives. Options for Phase II:

1. Keep as-is and discuss limitation; or  
2. Reformulate with explicit constraints for CTAEA only — **rejected** for primary fairness (would give CTAEA a different problem); or  
3. Add a **controlled ablation** later with explicit constraints for all constraint-capable methods under a separate experiment ID.

Primary fairness prefers (1) or a **shared** constrained reformulation in a dedicated experiment, never CTAEA-only.

---

## 14. Sign-off

| Role | Statement |
|------|-----------|
| Production PRIMARY | Conditionally fair if residual MA/Greedy eval overshoot is reported |
| Legacy DEBUG CSV | **Unfair — quarantine** |
| Exact MILP | Not an EA competitor; candidate_limit disclosed |

Update this file when instrumentation changes evaluation accounting.
