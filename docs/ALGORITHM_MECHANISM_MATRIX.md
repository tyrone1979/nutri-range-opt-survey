# ALGORITHM_MECHANISM_MATRIX.md — NutriOpt-Bench (TEVC)

**Purpose:** Support mechanistic Discussion: *which search mechanisms interact with RCFA regimes?*  
**Ground truth:** `baseline_algorithms.py`, `simple_baselines.py`, `exact_milp.py`, `run_benchmark.py`  
**Date:** 2026-09-08

NutriSwap-GA is **out of the primary matrix** (legacy protocol); see §9.

---

## 1. Mechanism families

| Family | Code | Algorithms in stack |
|--------|------|---------------------|
| A Pareto-dominance | DOM | NSGA2 |
| B Reference-vector / direction | REF | RVEA; NSGA3 (wired, optional primary) |
| C Decomposition | DEC | MOEAD (wired, optional primary) |
| D Indicator / archive | IND | SPEA2 |
| E Constraint-oriented / feasibility-biased | CSTR | CTAEA; BMOEA; ACMO-Diet |
| F Memetic / local improvement | MEM | MA |
| G Non-evolutionary baselines | BASE | Random; Greedy; RandomRepair |
| H Exact | EXACT | MILP (PuLP+CBC), small \(N\) / candidate subset |

---

## 2. Master mechanism matrix

| Algo | Family | Representation | Initialization | Selection | Variation | Replacement / elitism | Archive | Constraint handling | Repair | Diversity preservation | Objective path | Notes |
|------|--------|----------------|----------------|-----------|-----------|------------------------|---------|----------------------|--------|-------------------------|----------------|-------|
| BMOEA | CSTR | Variable-length index set | Uniform random cardinality in \([K_{\min},K_{\max}]\) | Prefer feasibles; fill rest by fitness-proportional infeasibles | Set crossover (union subsample); add/del/replace mutate | Feasible-first rebuild of population | None | Scalar penalty fitness; hard feas. for metrics | Implicit via unique/cardinality | Soft \(P_{\mathrm{div}}\) in fitness only | Native scalar \(f\) | Feasibility-biased EA |
| ACMO-Diet | CSTR | Variable-length index set | 70% meal-aware `random_individual`, else uniform | Keep best; tournament parents | Partial gene replace + mutate | Generational with elite best | None | Penalty + meal-aware init | Cardinality clip | Soft diversity in fitness | Native scalar \(f\) | Init bias toward meal cover |
| MA | MEM | Variable-length index set | Uniform random | Feasible-first like BMOEA | Crossover/mutate + intermittent LS toward target center | Feasible-first + LS updates | Best-feasible memory | Penalty + LS on nutrient center | LS replace | Soft diversity | Native scalar \(f\) | Extra LS evals (fairness note) |
| NSGA2 | DOM | Fixed pad length \(K_{\max}\), ints/−1 | `IntSampling` | Non-dominated sorting + crowding | SBX + PM + RoundingRepair | Elitist merge | None (pop only) | Soft via \((f_1,f_2)\); `n_constr=0` | RoundingRepair | Crowding distance | pymoo bi-obj | Classic dominance MOEA |
| RVEA | REF | Same padded | Same | Angle-penalized distance to ref vectors | SBX+PM+Rounding | Ref-vector environment selection | Ref dirs | Soft bi-obj | RoundingRepair | Ref-vector niche | pymoo bi-obj | May struggle if feasible region tiny |
| NSGA3 | REF | Same | Same | Reference-point niching | SBX+PM+Rounding | Ref-point selection | Ref dirs | Soft bi-obj | RoundingRepair | Ref-point niche | pymoo bi-obj | Optional primary |
| MOEAD | DEC | Same | Same | Neighbor Tchebycheff update | SBX+PM+Rounding | Neighbor replacement (nr) | Weight vectors | Soft bi-obj | RoundingRepair | Weight diversity | pymoo bi-obj | Optional primary |
| CTAEA | CSTR | Same | Same | Two-archive constrained MOEA logic | SBX+PM+Rounding | CA/DA style archives (pymoo) | Yes (algorithm-internal) | Designed for constraints but problem sets `n_constr=0` | RoundingRepair | Archive maintenance | pymoo bi-obj | **Mechanism mismatch risk:** constrained algo on unconstrained encoding |
| SPEA2 | IND | Same | Same | Strength + density | SBX+PM+Rounding | Archive truncation | External archive | Soft bi-obj | RoundingRepair | k-NN density in SPEA2 | pymoo bi-obj | Indicator/archive family |
| Random | BASE | Variable set | iid random subsets | None | Resample | Keep best-so-far pop by score | None | Evaluation only | None | None | Scalar eval | Lower bound |
| Greedy | BASE | Variable set | Meal seeds then add | Greedy min violation | Constructive add | Build pop of constructions | None | Constructive feasibility push | None | None | Violation-driven | Strong non-EA baseline |
| RandomRepair | BASE | Variable set | Random then repair | None | Delete high upper violators / add for deficits | Pop of repaired | None | Explicit repair | Yes | None | Scalar eval | Isolates repair mechanism |
| MILP | EXACT | Explicit binary \(x\) | Solver | Branch-and-cut | — | — | — | Hard constraints + slacks | — | — | Min total slack | Difficulty calibration |

---

## 3. Complexity / scalability characteristics (qualitative)

| Algo | Per-generation cost driver | Expected weak regime (hypothesis — test, do not assert) |
|------|----------------------------|--------------------------------------------------------|
| BMOEA / ACMO | \(O(P\cdot K\cdot m)\) eval | Very sparse + no repair |
| MA | Eval + LS candidate scans | Large \(N\) if LS candidate sets grow |
| NSGA2 | Nondominated sort \(O(M P^2)\) | Sparse feasible; many infeasible dominate poorly |
| RVEA / NSGA3 | Ref association | Sparse / highly discrete feasible pockets |
| MOEAD | Neighbor updates | Sparse feasible breaks aggregation assumptions |
| CTAEA | Archive maintenance | Soft-constraint encoding may underuse CHT |
| SPEA2 | Archive density | Scale of archive ops with \(P\) |
| Random | Pure sampling | Sparse feasible density \(\to 0\) |
| Greedy | Candidate scoring | Local optima / meal-infeasible traps |
| MILP | CBC search tree | Large \(N\) timeout |

---

## 4. How mechanisms map to TEVC questions

| Scientific question | Mechanisms to compare |
|---------------------|------------------------|
| Feasibility discovery under sparsity | CSTR vs DOM/REF/DEC vs Repair vs Random |
| Scaling with \(N\) | BASE vs MEM vs DOM; runtime vs FHR |
| Constraint dimensionality | All families; violation trajectories |
| Diversity among feasibles | SPEA2 crowding/archive vs BMOEA soft \(P_{\mathrm{div}}\) vs NSGA2 crowding |
| Role of repair | RandomRepair ablation vs Random; ACMO init |
| Role of local search | MA vs BMOEA |
| Role of explicit CHT | CTAEA vs NSGA2 under same bi-obj encoding |

---

## 5. Primary vs optional inclusion (TEVC main table)

**Primary (configs/e1_full.yaml):**  
BMOEA, ACMO-Diet, MA, NSGA2, RVEA, CTAEA, SPEA2, Random, Greedy, RandomRepair

**Recommended add if compute allows:** MOEAD, NSGA3 (paradigm coverage for DEC/REF)

**Calibration only:** MILP

**Exclude from main / supplement only:** NutriSwap-GA

---

## 6. Ablation targets (focused, not exhaustive)

| Ablation | Target algo | Mechanism isolated |
|----------|-------------|--------------------|
| Meal constraint on/off | BMOEA, NSGA2 | Structural hard constraint |
| Diversity penalty on/off | BMOEA | Soft diversity pressure |
| Repair on/off | Random vs RandomRepair | Repair |
| LS on/off | MA vs BMOEA | Memetic improvement |
| Meal-aware init on/off | ACMO vs BMOEA | Initialization bias |

---

## 7. Metrics each mechanism should be judged on

Always report jointly: **FHR, FFG_evals, FSC, BestFit, Diversity/Contrib, CV_nut, n_evals, runtime**.  
Dynamics (when logged): feasible fraction, best \(V_{\mathrm{nut}}\), diversity vs evaluations.

---

## 8. Implementation caveats (must appear in fairness discussion)

1. Native scalar vs pymoo bi-objective = different selection operators under shared hard feasibility.  
2. CTAEA run with `n_constr=0` may not exercise its constrained archives as designed.  
3. MA and Greedy can slightly exceed nominal eval budgets due to inner candidate scoring — record actual `n_evals`.  
4. Category diversity uses `food_id` digit prefix; NHANES sampling uses a different category field for stratification — do not confuse the two.

---

## 9. NutriSwap-GA

Present in `_nutriswap_ga.py` / `bench_others.py` under a **different** evaluation protocol.  
**Not** part of the primary mechanism matrix for TEVC. If reported, place in supplementary with explicit non-comparability.
