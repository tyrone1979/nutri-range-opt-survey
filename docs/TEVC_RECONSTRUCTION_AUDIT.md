# TEVC Reconstruction Audit — NutriOpt-Bench Phase II

**Audit date:** 2026-09-07  
**Repository:** `/home/zhaolei/nutri-range-opt-survey`  
**Target venue:** IEEE Transactions on Evolutionary Computation (TEVC)  
**Rule:** Implementation and result files are ground truth. Debug/pilot artifacts are not primary evidence.

---

## Executive verdict (pre-TEVC)

| Dimension | Status |
|-----------|--------|
| Infrastructure for a TEVC-oriented study | **Substantial (Phase I)** |
| Primary E1/E2/E3 experiments (12×10, fair budget) | **Not completed** |
| Mechanistic / trajectory / interaction analysis | **Mostly missing** |
| Manuscript framing for TEVC | **Too application-led; needs EC-first rewrite** |
| Current TEVC readiness (preliminary) | **~45–55 / 100 — do not submit** |

Phase II must **not restart** the project. It must **reframe, complete essential experiments, and deepen mechanism analysis**.

---

## 1. What has already been implemented?

### Core optimization stack
| Component | Path | Status |
|-----------|------|--------|
| Evaluation, diversity, data load | `common.py` | Executable |
| BMOEA, ACMO-Diet, MA | `baseline_algorithms.py` | Executable; production early-stop optional |
| pymoo NSGA2/RVEA/CTAEA/SPEA2 (+ NSGA3/MOEAD wired) | `baseline_algorithms.py` | Executable via `run_pymoo_algorithm` |
| FoodSpace / Target / legacy E1–E4 runners | `e1_scaling.py` … `e4_alignment.py` | Executable but **legacy budget wiring** |
| Fair production runner | `run_benchmark.py` + `benchmark_core.py` | Executable |
| Random / Greedy / RandomRepair | `simple_baselines.py` | Executable |
| MILP/CBC calibration | `exact_milp.py`, `run_exact.py` | Executable (candidate-subset) |
| Target clustering / stratification | `target_analysis*.py` | Executable |
| Target coverage PCA | `analyze_target_coverage.py` | Done |
| Data validation | `validate_data.py` | Done |
| Statistical helpers | `statistical_tests.py` | Code present; **not run on primary data** |
| NutriSwap + alternate MOEAs | `_nutriswap_ga.py`, `bench_others.py` | Legacy / separate protocol |

### Documentation & literature (Phase I)
- `docs/PROJECT_AUDIT.md`, `PROBLEM_FORMULATION.md`, `BENCHMARK_PROTOCOL.md`, `METRIC_DEFINITIONS.md`, `DATA_VALIDATION.md`, `CLAIM_EVIDENCE_MATRIX.md`, `FINAL_REVIEW.md`
- `literature/` search protocol, taxonomy, **seed** matrix (not full PRISMA)
- `REPRODUCIBILITY.md`, `README.md`
- Production configs under `configs/`
- Manuscript: `manuscript/nutriopt_bench.{tex,pdf,docx}` + supplementary

### Missing for TEVC Phase II (explicit gaps)
- `docs/ALGORITHM_MECHANISM_MATRIX.md`
- `docs/FAIRNESS_AUDIT.md`
- `docs/TEVC_CLAIM_EVIDENCE_MATRIX.md`
- `docs/TEVC_FINAL_REVIEW.md`
- `docs/TEVC_READINESS_REPORT.md`
- Feasible-density estimator; distance-to-feasibility logging
- Per-evaluation trajectory recorder + plots
- Algorithm × regime interaction analysis pipeline
- Focused mechanism ablations (runner config exists; **no ablation results**)
- Full PRISMA literature screening counts
- E2/E3/E4 primary result CSVs
- Critical-difference / rank-stability figures from primary data

---

## 2. Which experiments are actually complete?

| Experiment | Complete? | Evidence |
|------------|-----------|----------|
| Data validation | **Yes** | `results/reproducibility/data_validation.json`, `docs/DATA_VALIDATION.md` |
| Target coverage (descriptive) | **Yes** | `results/target_coverage.csv`, `results/figures/fig_target_coverage.*` |
| Exact MILP small calibration | **Yes (limited)** | `results/raw/exact_small.csv` — 12 reps, candidate_limit=300, NHANES-500 |
| Profile (Stage 1) | **Yes** | `e1_profile.csv` — 1 tgt × 1 rep × 6 algos, 400 evals |
| Mini-pilot (Stage 2 partial) | **Yes** | `e1_minipilot.csv` — 4 tgt × 2 rep × 7 algos, 2400 evals |
| E1 full scaling | **No** | No `e1_scaling_full.csv` |
| E2 sparsity | **No** | No `e2_*.csv` primary |
| E3 dimension | **No** | No `e3_*.csv` primary |
| E4 cross-regime | **No** | No inputs to meta-analysis |
| Ablations | **No** | Config only |
| Search dynamics trajectories | **No** | Not implemented as first-class logging |
| Full statistical battery on primary | **No** | Helpers unused on PRIMARY |

---

## 3. Which experiments are only debug-level?

| Artifact | Why debug / non-TEVC |
|----------|----------------------|
| `results/raw/e1_scaling.csv` (154 rows) | `targets_total≈2` NHANES + 6 small; **repeats=1**; **unequal n_evals** (legacy early stop); config budgets not wired through old `e1_scaling.ALGO_RUNNERS` |
| `config.yaml` / `config_debug.yaml` | Explicitly debug (2 targets, 1 repeat) |
| `output/experiment_report.txt`, `DEEP_ANALYSIS.txt`, `experiment_results.json` | NutriSwap-centric legacy narrative; different protocol |
| `e1_profile.csv` | Engineering PROFILE only |
| `e1_minipilot.csv` | PILOT: 4 targets, 2 seeds — **insufficient for claims** |

**Policy:** Cite only rows with `result_class ∈ {PRIMARY, ROBUSTNESS, EXACT}` (or equivalent) after production runs. PROFILE/PILOT/DEBUG are development artifacts.

---

## 4. Which algorithms are genuinely implemented and executable?

| Algorithm | Implementation | Callable from `run_benchmark.py` | In `e1_full.yaml` |
|-----------|----------------|----------------------------------|-------------------|
| BMOEA | Native | Yes | Yes |
| ACMO-Diet | Native | Yes | Yes |
| MA | Native | Yes | Yes |
| NSGA2 | pymoo | Yes | Yes |
| RVEA | pymoo | Yes | Yes |
| CTAEA | pymoo | Yes | Yes |
| SPEA2 | pymoo | Yes | Yes |
| Random | `simple_baselines` | Yes | Yes |
| Greedy | `simple_baselines` | Yes | Yes |
| RandomRepair | `simple_baselines` | Yes | Yes |
| NSGA3 | pymoo branch | **Yes (code path)** | **No** |
| MOEAD | pymoo branch | **Yes (code path)** | **No** |
| MILP/CBC | PuLP | Via `run_exact.py` | Separate config |
| NutriSwap-GA | `_nutriswap_ga` / `bench_others` | **Not** in production runner | **Exclude from main** |

---

## 5. Which algorithms are merely mentioned but not implemented?

| Name | Situation |
|------|-----------|
| Local Search (standalone baseline) | Only as **subroutine inside MA**; no standalone LS baseline class for fair budget comparison |
| NSGA-III / MOEA/D | Implemented in pymoo adapter but **not in primary configs**; also reimplemented in legacy `bench_others.py` under different protocol |
| AGEMOEA / SMSEMOA | Imported/optional in pymoo adapter; not in NutriOpt-Bench primary set |
| “Constraint handling ON/OFF ablations” | Mentioned in protocol; **not executed** |
| Feasible-density / distance-to-feasibility metrics | Defined conceptually; **not computed in runners** |

---

## 6. Which results are reproducible?

| Result | Reproducible command | Notes |
|--------|----------------------|-------|
| Data validation | `python3 validate_data.py` | Deterministic |
| Target coverage | `python3 analyze_target_coverage.py` | Seeded PCA/NN |
| Exact small | `python3 run_exact.py configs/exact_small.yaml` | CBC; seed in config; **candidate subsample** affects feasibility |
| Profile / minipilot | `run_benchmark.py` + configs | Seeds via `make_seed`; fair evals |
| Legacy `e1_scaling.csv` | Old `e1_scaling.py --config config.yaml` | Reproducible as **unfair debug**, not as TEVC evidence |
| Manuscript numbers from PRIMARY | N/A | PRIMARY not generated |

Env snapshot: `results/reproducibility/env.json`.

---

## 7. Which results use insufficient repeats?

| Dataset | Repeats | TEVC minimum | Verdict |
|---------|---------|--------------|---------|
| Legacy E1 | 1 | 10 (prefer 20) | Fail |
| Profile | 1 | 10 | Fail (OK for profiling) |
| Mini-pilot | 2 | 10 | Fail |
| Exact | 1 solve / target | N/A (deterministic given candidates) | OK for calibration |
| PRIMARY E1–E3 | 0 completed | 10–20 | Pending |

---

## 8. Which results use unequal computational budgets?

| Artifact | Equal budget? | Detail |
|----------|---------------|--------|
| Legacy `e1_scaling.csv` | **No** | Early stop at ≥20 feasibles; native vs pymoo different effective evals; median n_evals from 2k (ACMO/BMOEA) to 100k (CTAEA) |
| Old `e1_scaling.run_*` | **No** | Does not pass YAML `max_evaluations` / `population_size` into algorithms |
| `run_benchmark.py` PROFILE/PILOT | **Yes (intended)** | Shared `max_evaluations`; early_stop=null; minor overshoot on MA/Greedy (local evals / candidate scoring) — document as secondary sensitivity |
| pymoo path historically | Seed fixed to 42 | **Fixed in production runner** via `seed=` argument |

**TEVC action:** Deprecate legacy runner for any reported table; all PRIMARY through `run_benchmark.py`.

---

## 9. Which results are statistically defensible?

| Analysis | Defensible now? |
|----------|-----------------|
| Friedman / Nemenyi / Wilcoxon / effect sizes on PRIMARY | **No** — no PRIMARY matrix |
| Descriptive MILP feasibility rates (8/12 feasible under candidate limit) | **Yes**, with caveat on candidate_limit |
| Target-coverage NN distances (cluster-12 vs random-12) | **Yes** as descriptive coverage proxy, not algorithm ranking |
| Mini-pilot mean FHR by algorithm | **No** for inferential claims (n_seeds=2, n_targets=4) |
| Legacy NutriSwap “significantly better” narrative | **No** |

---

## 10. Which claims currently appearing in the manuscript are unsupported?

From `manuscript/nutriopt_bench.tex` + keyword scan:

| Claim / tone | Status |
|--------------|--------|
| Full algorithm ranking / “best algorithm” | Manuscript correctly **withholds** — keep |
| NutriOpt-Bench as complete empirical study of scale/sparsity/dimension | **Overstates** relative to completed experiments — rewrite as “protocol + provisional calibration” until PRIMARY exists |
| Systematic survey completeness | Seed matrix only — **cannot claim finished PRISMA** |
| “First” / “novel” / “state-of-the-art” | Avoid; manuscript still contains soft “novel”-adjacent framing in places — scrub in Phase II rewrite |
| NutriSwap mention | Appropriately de-centered, but **Introduction still opens with nutrition motivation** — wrong for TEVC |
| Regime-dependent EA performance as established finding | **Hypothesis only** — must not appear as Conclusion until E1–E4 support it |
| Clinical / health effectiveness | Correctly limited — keep |

---

## 11. Which experiments are essential for TEVC?

**Must have (CRITICAL):**

1. **E2 Feasible-region sparsity** with feasible-density estimation + FHR/FFG vs density (mechanism centerpiece).  
2. **E1 Scaling** on NHANES samples under fair budget (separate recipe catalogs; do not pool). Prefer N∈{5K,10K,25K,50K,100K} (+ full if feasible).  
3. **E3 Constraint dimensionality** {6,9,12,16} under fair budget.  
4. **E4 Cross-regime algorithm analysis** (rank stability + algorithm×regime map) — rename from “Alignment”.  
5. **Equal max-evaluations protocol** + fairness audit document.  
6. **Simple baselines + MILP difficulty calibration**.  
7. **≥10 seeds × 12 representative targets** primary; robustness 45×5 if possible.  
8. **Search dynamics trajectories** for ≥2–3 representative algos × sparse/dense cases.  
9. **Focused ablations** (constraint handling / repair / diversity) on 1–2 mechanisms.  
10. **Mechanism-oriented algorithm taxonomy** feeding Discussion.  
11. **EC-first manuscript rewrite** answering RQ2–RQ6 with Experiment→Metric→Statistic→Figure→Claim.

---

## 12. Which experiments are optional?

| Item | Why optional |
|------|--------------|
| N≈210K full NHANES for all algos | Costly; include subset of algos or timeout reporting |
| Extra sparsity grid {0.25…2.0} | Nice if budget allows; {0.5,1.0,1.5} is minimum |
| 20 seeds | Prefer for headline comparisons; 10 acceptable if reported |
| 100-target extended validation | Robustness, not core mechanism |
| NSGA-III / MOEA/D in main table | Valuable paradigm coverage; add if compute allows |
| NutriSwap as extra baseline | Optional; risk of bias perception — prefer supplement or omit |
| LLM+optimization / multi-day planning | Future work only |
| Full clinical unit validation | Domain appendix, not TEVC core |

---

## 13. Which experiments should be removed (from main TEVC story)?

| Remove / demote | Reason |
|-----------------|--------|
| Legacy NutriSwap leaderboard narrative (`output/*`) | Algorithm advertisement; unfair/different protocol |
| Mixing Foods.com/Allrecipes curves with NHANES as one “scaling law” | Different data-generating process, dims, meal constraints |
| DEBUG `e1_scaling.csv` tables in main text | Unequal budgets |
| Application-heavy “healthy diet” background | Dilutes EC contribution |
| Claiming meal/preference/cost/sustainability as core RCFA constraints without systematic experiments | Overloads formulation |
| Isolated BestFit-only comparisons | Feasibility must be first-class |

---

## 14. Which parts of the current manuscript are too application-oriented?

| Section / content | Issue | TEVC action |
|-------------------|-------|-------------|
| Title emphasizing “Diet Assembly” first | Reads as domain survey | Prefer EC/combinatorial-search-first title |
| §1 Motivation opens with personalized nutrition | Application lead | Start from binary constrained combinatorial search + sparse feasibility |
| Heavy diet-planning survey framing without EA mechanism thread | Survey-of-apps feel | Keep taxonomy but pivot to **problem regime for EAs** |
| Practical guide phrased as nutrition tool choice | Domain tone | Reframe as **algorithm×regime selection for EC practitioners** |
| Lengthy dataset nutrition discussion | Compress | Move units/DRI caveats to supplement |
| Any residual NutriSwap spotlight | Bias risk | Keep out of main narrative |

---

## 15. Which parts are insufficiently connected to evolutionary computation theory/mechanism?

| Gap | Why it matters for TEVC |
|-----|-------------------------|
| No algorithm **mechanism matrix** (dominance / decomposition / reference / repair / penalty) | Reviewers need mechanism→behavior links |
| No **feasible-region density** vs performance analysis | Core scientific question unanswered |
| No **search dynamics** (feasible fraction / violation / diversity vs evals) | Static tables ≠ TEVC insight |
| No **algorithm × scale/sparsity/dimension interaction** tests | Central claim untested |
| Constraint handling treated as scalar penalty only | Feasibility not first-class beyond FHR label |
| pymoo bi-objective encoding (`f1` nutrient+diversity, `f2` −meal) vs native scalar fitness | Mechanism inequivalence under-discussed |
| Diversity soft penalties vs hard feasibility | Must clarify selection pressure differences |
| Early-stop history / MA local-search extra evals | Fairness & complexity undertheorized |
| Literature seed set thin on constrained MOEA / sparse feasible-region EC | Need EC-side citations (constraint handling, CHT, sparse Pareto) beyond diet papers |

---

## Algorithm readiness snapshot (for later matrix)

| Family | Algorithms present | Mechanism study ready? |
|--------|--------------------|------------------------|
| A Dominance | NSGA-II | Yes (code) |
| B Reference-vector | RVEA (+ NSGA-III optional) | Partial |
| C Decomposition | MOEA/D (code, not primary) | Gap if omitted |
| D Indicator/archive | SPEA2 | Yes |
| E Constraint-specialized | CTAEA, BMOEA, ACMO-Diet | Yes; need ablations |
| F Simple | Random, Greedy, Repair | Yes; LS standalone missing |
| G Exact | MILP | Yes (small N) |
| Memetic | MA | Yes; LS coupled |

---

## Manuscript component triage (subtraction preview)

| Component | Tag |
|-----------|-----|
| Problem formulation matching code | **KEEP** (rename RCFA; tighten core vs optional constraints) |
| Fair budget protocol | **KEEP** |
| E1–E3 design | **KEEP** (execute properly) |
| E4 “Alignment” | **REWRITE** → Cross-Regime Algorithm Analysis |
| Seed literature protocol | **KEEP**; expand EC literature |
| Current Intro nutrition lead | **REWRITE** |
| Pilot/profile tables in main as “results” | **COMPRESS** / move to supplement as calibration |
| Legacy NutriSwap results | **DELETE** from main |
| Mechanism discussion | **MUST ADD** |
| Trajectory figures | **MUST ADD** |
| Algorithm×regime map | **MUST ADD** (fill after data) |
| Full PRISMA counts | **MUST ADD** before submission |
| Claim–evidence TEVC matrix | **MUST ADD** |

---

## Priority order for Phase II (post-audit)

1. Freeze RCFA core formulation + `ALGORITHM_MECHANISM_MATRIX.md` + `FAIRNESS_AUDIT.md`.  
2. Implement feasible-density + trajectory logging in production runner.  
3. Add MOEA/D (and optionally NSGA-III) to primary configs if compute allows.  
4. Run E2 (sparsity) and E1 (scaling) as first PRIMARY campaigns; then E3.  
5. Statistical + E4 cross-regime analysis; ablations; dynamics figures.  
6. **Only then** rewrite manuscript EC-first; fill claim–evidence matrix; TEVC readiness report.

---

## Bottom line

The repository is a **credible TEVC prototype**: fair runner, baselines, MILP calibration, target protocol, and honest Phase I manuscript caveats.

It is **not** yet a TEVC submission:

- PRIMARY experiments absent;  
- mechanism/dynamics/interaction analysis absent;  
- manuscript still diet-survey flavored;  
- legacy unequal-budget CSVs must stay quarantined.

**Next allowed step after this audit:** freeze formulation/mechanism/fairness docs and instrumentation — **not** final scientific conclusions, and not manuscript rewriting until this audit is accepted as the baseline (this file fulfills STEP 1).
