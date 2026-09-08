# NutriOpt-Bench Repository Audit

**Audit date:** 2026-09-07  
**Scope:** All Python modules, configs, datasets, results, figures, documentation  
**Rule:** Implementation behavior is ground truth; comments that contradict code are flagged.

---

## A. Current scientific question

The code solves **range-constrained combinatorial food-subset assembly**:

> Select a variable-cardinality subset of candidate food items (indices into a nutritional matrix) such that the **sum** of selected items’ nutrient vectors lies inside per-nutrient intervals \([L_j, U_j]\), optionally covering breakfast/lunch/dinner, while soft-penalizing low food-category and nutrient-contribution diversity.

This is **not** continuous portion optimization, calorie-only planning, or ranking-based food recommendation. Decision variables are discrete food indices (subset / padded integer encoding), not continuous grams.

Related but separate code paths:
- `bench_others.py` / `_nutriswap_ga.py`: older NutriSwap-GA vs MOEA comparison on recipe-style data (algorithm-centric; **not** the NutriOpt-Bench survey framing).
- `bench_nhanes.py`: sequential per-target runner for BMOEA/ACMO/MA/pymoo (legacy).
- `target_analysis*.py`: cluster/stratify the 4019 targets into representatives.

---

## B. Mathematical formulation (as implemented)

### Decision variables

| Aspect | Implementation truth |
|--------|----------------------|
| Representation (native EA) | Variable-length vector of **integer food indices** into the food matrix; duplicates removed via `np.unique` in several operators |
| Representation (pymoo) | Fixed-length integer string of length `max_foods`; entries in \([-1, n_{\mathrm{valid}}-1]\); negative = unused slot (`DietProblem`) |
| Continuous portions | **Not used**. `grams` column in NHANES is **not** a decision variable; each row is treated as one selectable item with fixed nutrient amounts |
| Binary indicator \(x_i\in\{0,1\}\) | Implicit via set of indices; not solved as an explicit MILP in current code |

### Cardinality

| Setting | Source | Value |
|---------|--------|-------|
| Global defaults | `common.py` | `MIN_FOODS=3`, `MAX_FOODS=30` |
| NHANES FoodSpace | `e1_scaling.load_nhanes_full` | `min_foods=5`, `max_foods=30` |
| Allrecipes / Foods.com | loaders | `min_foods=3`, `max_foods=5` |
| Feasibility gate | `evaluate_batch` | if \(n\notin[\mathrm{min},\mathrm{max}]\), individual scores 0 / skipped |

### Nutritional constraints

For selected indices \(\mathcal{S}\), nutrient columns `col_idx`:

\[
t = \sum_{i\in\mathcal{S}} A_{i,\mathrm{col}},\quad
\text{nutrient\_penalty}=\sum_j \max(0,L_j-t_j)+\max(0,t_j-U_j)
\]

Nutrient-feasible iff `nutrient_penalty < 1e-6`.

### Meal constraints

If `has_meal_constraint=True` (NHANES path):

- `meal_tags` ∈ {0 breakfast, 1 lunch, 2 dinner}
- `meal_coverage` = number of distinct tags present among selected foods
- `meal_penalty = (3 - meal_coverage) * MEAL_COVERAGE_WEIGHT` with `MEAL_COVERAGE_WEIGHT=100`
- Meal-feasible iff `meal_coverage == 3`

If False (recipe datasets): meal penalty = 0, meal treated as feasible.

### Diversity (soft penalties, not hard constraints)

1. **Category diversity:** `food_id` string prefix (first 2 digits) → `#categories / max_categories`;  
   `diversity_penalty = (1 - score) * DIVERSITY_WEIGHT` (`DIVERSITY_WEIGHT=10`).
2. **Contribution diversity:** weighted mix of coverage, entropy, complementarity, dominant nutrients (`compute_contribution_diversity`);  
   `contrib_penalty = (1 - contrib) * DIVERSITY_WEIGHT * 0.5`.

### Objective / fitness (native algorithms)

\[
\text{total\_penalty} = \text{nutrient} + \text{meal} + \text{diversity} + \text{contrib}
\]
\[
\text{fitness} = \frac{1}{1 + \max(0,\text{total\_penalty})}
\]

**Feasible** := nutrient_feasible ∧ meal_feasible  
(Diversity does **not** enter the hard feasibility definition.)

### pymoo bi-objective encoding (`DietProblem`)

- \(f_1\) = nutrient_penalty + diversity_penalty  
- \(f_2\) = −meal_coverage (or −3 if meal constraint off)  
- `n_constr=0` (constraints folded into objectives / penalties)

### Stopping criteria (native)

Defaults from `common.py` (used when runners do not override):

| Parameter | `common.py` | `config.yaml` | `e1` pymoo wrapper |
|-----------|-------------|---------------|--------------------|
| Population | 2000 | 2000 | hard-coded 2000 |
| Generations | **50** | **20** | **50** |
| Max evaluations | **500000** | **100000** | generations-driven |

Native loops: stop if `n_evals >= max_evals` **or** `#feasible ≥ 20` (early stop) **or** generations exhausted.

**CRITICAL INCONSISTENCY:** `config.yaml` budgets are **not** passed into `run_bmoea` / `run_acmo` / `run_ma` / `run_pymoo`. Algorithms use `common.py` defaults (or pymoo hardcodes). Early stop at 20 feasibles further breaks equal-budget comparison. Observed `n_evals` in existing results ranges from 2000 to 100000.

### Random seeds

- `run_single`: `run_seed = (seed * 10000 + scale + rep) % 2**32-1`
- pymoo `algorithm.setup(..., seed=42)` — **fixed 42**, largely ignoring per-run seed (**fairness bug**).

### Prefiltering

`prefilter` / `bench_nhanes`: discard foods with any nutrient > upper bound. E1 runners pass **full** `food_ids`, not prefiltered sets (prefilter used in legacy bench path).

---

## C. Algorithm audit

| Algorithm | Family | Representation | Init | Variation | Constraint handling | Objectives | Pop / Term | Native vs pymoo | Seed |
|-----------|--------|----------------|------|-----------|---------------------|------------|------------|-----------------|------|
| BMOEA | Single-obj EA / feasibility-biased | Variable int set | Uniform random cardinality | Set crossover; add/del/replace mutate | Penalty fitness; elite keep feasibles | Scalar fitness | POP_SIZE / gens / max_evals / early≥20 | Native | `np.random.seed` before run |
| ACMO-Diet | Single-obj EA | Variable int set | 70% meal-aware `random_individual`, else random | Tournament; partial gene replace; mutate | Penalty; keep best | Scalar fitness | Same | Native | Same |
| MA | Memetic | Variable int set | Random | Crossover/mutate + local search toward target center | Penalty + LS | Scalar fitness | Same + LS evals | Native | Same |
| NSGA2 | Dominance MOEA | Fixed padded ints | `IntSampling` | SBX+PM+Rounding | Soft via \(f_1,f_2\) | Bi-obj | pop=2000, n_gen=50 | pymoo | setup seed=42 |
| RVEA | Reference-vector MOEA | Same | Same | Same | Soft | Bi-obj | Same | pymoo | seed=42 |
| CTAEA | Constrained MOEA (used unconstrained) | Same | Same | Same | Soft (`n_constr=0`) | Bi-obj | Same | pymoo | seed=42 |
| SPEA2 | Indicator/archive MOEA | Same | Same | Same | Soft | Bi-obj | Same | pymoo | seed=42 |
| NutriSwap-GA | Specialized GA (`bench_others`) | Recipe/meal encoding | Fingerprint-guided | Domain operators | Domain | Multi metrics | Separate protocol | Native (legacy) | Separate |
| Random / Greedy / MILP | — | — | — | — | — | — | — | **Not in E1–E3** | — |

NSGA-III / MOEA/D exist in `bench_others.py` and partially imported in `baseline_algorithms.py` but are **not** in `config.yaml` / `ALGO_RUNNERS`.

---

## D. Dataset audit

### `data/nhanes.csv`

| Property | Value |
|----------|-------|
| Rows | 213,569 |
| Unique `food_id` | 9,640 |
| Duplicate IDs | 203,929 row-level duplicates (same ID, different grams/nutrients — **portion instances**, not unique foods) |
| Columns | food_id, grams, 16 nutrients |
| Nutrients | calorie, protein, carb, sugar, fiber, saturated_fat, cholesterol, folic_acid, vitamin_b12, vitamin_c, vitamin_d, calcium, phosphorus, potassium, iron, sodium |
| Missingness | vitamin_d ≈ 31.5%; others ≈ 0 |
| Units | Implicit dietary units (kcal, g, mg, µg — **not documented in code**) |
| Meal info | Via join to `food_type.csv` |
| Category | Derived from `food_id` digit prefix in diversity code; E1 also sets `categories = i // 1000` for sampling (**inconsistent category definitions**) |
| Normalization | None; missing nutrients filled with 0 after drop cols with >50% missing |
| After meal map | Rows without meal mapping dropped |

### `data/food_type.csv`

- 45,083 rows; 8,683 unique food_ids; `eating_type` ∈ {1…19, 91} mapped to breakfast/lunch/dinner via `EATING_TYPE_MAP`.

### `data/food_code.csv`

- 11,338 foods; descriptive metadata; not on critical E1 path.

### `data/food_data.csv` (labeled Allrecipes in config)

- 1,584 rows; columns calories, protein, carbs, fat (+ meta); loader drops non-positive; **4 nutrients**; no meal tags.

### `data/allrecipes.json` (labeled Foods.com in config)

- 1,798 recipes; nutrition parsed from nested dict; **4 nutrients**; no meal tags.  
- **Naming swap risk:** `load_allrecipes()` reads `food_data.csv`; `load_foods_com()` reads `allrecipes.json`.

### `data/target.csv`

- 4,019 targets; Python-dict strings of nutrient → `[lo, hi]`; dims 6–15 (mean ≈ 8.7).
- Representative files: 12 / 45 / 100 (clustered / stratified) — **E1 `load_targets` does not use these files**; it re-selects “widest” targets from full `target.csv` with `fixed_dim=6` for NHANES in E1.

### Small-dataset targets

`generate_small_dataset_targets`: random meals → P20–P80 style percentile intervals; **not** the 4019 NHANES target distribution.

---

## E. Existing result audit

### Files present

| Path | Status |
|------|--------|
| `results/raw/e1_scaling.csv` | 154 rows |
| `e2_sparsity.csv` | **Missing** |
| `e3_dimension.csv` | **Missing** |
| `results/e4_analysis` | **Missing** |
| `output/experiment_report.txt`, `DEEP_ANALYSIS.txt` | Legacy NutriSwap narrative |
| `output/experiment_results.json` | Legacy |
| `cluster_analysis.png` | Target clustering viz |
| README / LaTeX manuscript | **Absent** at audit time |

### E1 content (flagged)

| Check | Observation | Flag |
|-------|-------------|------|
| targets_total | config=2; NHANES runs use 2 targets; small DS use 6 generated | **DEBUG_RESULT** |
| repeats | Only `rep=0` | **DEBUG_RESULT** |
| Algorithms | All 7 present | OK for coverage |
| E2/E3 | Not run | **DEBUG_RESULT** |
| Evaluation budget | `n_evals` 2k–100k; early stop; config unused | **DEBUG_RESULT** (unfair) |
| Seeds | pymoo fixed 42 | **DEBUG_RESULT** |
| Mixed data mechanisms | Recipe 4-D vs NHANES 6-D+meal | Must not pool as one curve |
| FHR≈1.0 for most | Likely easy targets / early stop / small budget artifacts | Unsupported for publication claims |
| NutriSwap reports in `output/` | Different protocol/dataset | Do **not** cite as NutriOpt-Bench evidence |

### Unsupported conclusions (must not enter manuscript as fact)

- Any claim that algorithm X is best on NutriOpt-Bench.
- Scaling laws, sparsity phase transitions, dimension curses from current CSVs.
- NutriSwap superiority from `output/` as evidence for this survey+benchmark paper.

---

## F. Code / documentation inventory

| Module | Lines | Role |
|--------|------:|------|
| `common.py` | 283 | Load, evaluate, diversity metrics |
| `baseline_algorithms.py` | 756 | BMOEA, ACMO, MA, pymoo adapter |
| `e1_scaling.py` | 606 | Core benchmark runner |
| `e2_sparsity.py` | 250 | Interval width factors |
| `e3_dimension.py` | 246 | Dim projection |
| `e4_alignment.py` | 421 | Meta-analysis (needs E1–E3) |
| `bench_others.py` | 1427 | Legacy multi-algo bench |
| `_nutriswap_ga.py` | 308 | NutriSwap |
| `bench_nhanes.py` | 152 | Legacy NHANES loop |
| `target_analysis.py` | 689 | 12-cluster representatives |
| `target_analysis_stratified.py` | 497 | Stratified 100 |
| `statistical_tests.py` | 185 | Friedman/Wilcoxon/Nemenyi/Cohen |
| `config.yaml` | 70 | **Debug** settings |

---

## G. Critical issues to fix before production runs

1. **Wire config budgets** into all algorithm runners; disable or standardize early stopping for fair comparison.
2. **Pass per-run seeds** into pymoo.
3. **Use documented representative targets** (`representative_targets*.csv`) instead of ad-hoc wide-target sampling alone; report coverage.
4. **Separate recipe vs NHANES** analyses.
5. **Unify category definitions** (digit prefix vs `i//1000`).
6. **Add Random / Greedy / MILP** baselines.
7. **Preserve `config.yaml` as debug**; add `configs/*_full.yaml`.
8. Treat all current numerical results as **DEBUG_RESULT** only.

---

## H. Audit verdict

The repository is a **viable prototype** of a range-constrained diet-assembly benchmark, but it is **not publication-ready**:

- Scientific question and core evaluation logic are clear.
- Fairness protocol is broken (budget, seed, early stop).
- Existing results are debugging-only.
- Survey/taxonomy/manuscript infrastructure was missing at audit time.

**Next step (STEP 2+):** freeze formulation & protocol documents; build literature infrastructure without fabricating citations; implement production configs and fair runners before any claimed experimental conclusions.
