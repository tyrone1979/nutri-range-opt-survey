# RCFA Formulation (TEVC Freeze) — Range-Constrained Food Assembly

**Status:** FROZEN for Phase II TEVC reconstruction (2026-09-08)  
**Ground truth:** `common.py`, `exact_milp.py`, `run_benchmark.py`  
**Naming:** Manuscript/TEVC use **RCFA**. Legacy docs may say RCDA; they denote the same core problem.

This document freezes the **general evolutionary-computation problem regime**.  
Diet/nutrition data instantiate the regime; they are not the intellectual center.

---

## 1. Positioning (what we claim / do not claim)

**Claim:** RCFA is a **unified benchmark/problem regime** combining:
binary decisions, combinatorial subset search, interval (box) constraints, optional structural constraints, and potentially sparse feasible regions at large \(N\).

**Do not claim:** that RCFA is an entirely new mathematical class invented by this paper, unless the completed systematic review proves a uniqueness claim (it currently does not).

---

## 2. Decision representation

\[
x \in \{0,1\}^N,\qquad
x_i = 1 \iff \text{food instance } i \text{ is selected}.
\]

Implementation note: native EAs often store a variable-length index set \(\mathcal{S}=\{i:x_i=1\}\) (with `np.unique`); pymoo uses a fixed-length padded integer encoding of length \(K_{\max}\). Both decode to the same binary selection semantics.

**Not in core RCFA:** continuous portion variables (grams as decisions). NHANES `grams` only define fixed nutrient content of each selectable row.

---

## 3. Nutrient totals and interval constraints (CORE)

Nutrient matrix \(A\in\mathbb{R}^{N\times m}\). For active nutrients \(j\in J\):

\[
z_j(x)=\sum_{i=1}^{N} A_{ij} x_i,
\qquad
L_j \le z_j(x) \le U_j \quad \forall j\in J.
\]

Nutrient violation:

\[
V_{\mathrm{nut}}(x)=\sum_{j\in J}\Big(\max(0,L_j-z_j)+\max(0,z_j-U_j)\Big).
\]

Nutrient-feasible iff \(V_{\mathrm{nut}}(x)<10^{-6}\) (implementation tolerance).

---

## 4. Food-count constraint (CORE)

\[
K_{\min} \le \sum_i x_i \le K_{\max}.
\]

| Setting | \(K_{\min}\) | \(K_{\max}\) |
|---------|-------------:|-------------:|
| NHANES FoodSpace (`e1_scaling` / production) | 5 | 30 |
| Recipe catalogs | 3 | 5 |
| `common.py` defaults | 3 | 30 |

---

## 5. Meal coverage (CORE when enabled; dataset-dependent)

For meal sets \(M_m\) (breakfast/lunch/dinner), when `has_meal_constraint=True`:

\[
\sum_{i\in M_m} x_i \ge 1 \quad \text{for each required meal } m\in\{0,1,2\}.
\]

Disabled for recipe datasets without meal tags.

---

## 6. Optional application-specific constraints (NOT core benchmark axes)

The following may appear in literature or future extensions but are **not** systematically varied in E1–E3 unless an ablation explicitly enables them:

- allergies / exclusions  
- budget / cost  
- preference scores  
- sustainability  
- multi-day linking / repetition limits  
- cultural rules  

**Diversity** (category / contribution) is implemented as a **soft penalty** in scalar fitness; it is **not** part of hard feasibility in the core definition.

---

## 7. Hard feasibility (first-class)

\[
\mathrm{Feasible}(x)
\iff
\bigl(K_{\min}\le \|x\|_0\le K_{\max}\bigr)
\land \bigl(V_{\mathrm{nut}}(x)<10^{-6}\bigr)
\land \bigl(\neg\mathrm{meal\_flag}\lor \text{all meals covered}\bigr).
\]

TEVC study separates:

1. **Feasibility discovery** (FHR, FFG / FFG_evals)  
2. **Feasible-region exploration** (FSC, diversity among feasibles)  
3. **Objective improvement among feasibles** (BestFit / soft penalties)  
4. **Diversity preservation** (category + contribution metrics)

Do not collapse (1)–(4) into a single BestFit ranking.

---

## 8. Objectives used by algorithms (implementation paths)

### 8.1 Native scalar path (BMOEA, ACMO-Diet, MA, Random/Greedy/Repair)

\[
P = V_{\mathrm{nut}} + P_{\mathrm{meal}} + P_{\mathrm{div}},
\qquad
f(x)=\frac{1}{1+\max(0,P)}\ \to\ \text{maximize}.
\]

### 8.2 pymoo bi-objective path (NSGA2, RVEA, CTAEA, SPEA2, …)

\[
\min f_1 = V_{\mathrm{nut}} + (1-d_{\mathrm{cat}})\cdot 10,
\qquad
\min f_2 = -C_{\mathrm{meal}}
\]

with `n_constr=0` (constraints folded into objectives).  
**Fairness note:** native and pymoo paths use different selection pressures; both share the same hard feasibility definition and evaluation budget. Document in fairness audit.

---

## 9. Structural characteristics of the RCFA regime (for EC)

| Property | RCFA manifestation |
|----------|-------------------|
| Representation | Binary / combinatorial subset |
| Constraint geometry | Interval (box) couplings via shared \(x\) |
| Feasible region | Often sparse under tight \([L,U]\) and meal cover |
| Decision dimension | \(N\) up to \(10^5\) food instances |
| Evaluation | Cheap linear aggregation; hardness is combinatorial |
| Multiobjective structure | Soft diversity / meal objectives optional |

---

## 10. Intrinsic infeasibility

If no \(x\) satisfies CORE constraints, the instance is **intrinsically infeasible**.  
Report via MILP/bounding pre-check. Do not attribute such failures to evolutionary algorithms.

---

## 11. Relation to Phase I `PROBLEM_FORMULATION.md`

This RCFA freeze **supersedes** naming and contribution framing for TEVC.  
Numeric tolerances, penalties, and \(K_{\min}/K_{\max}\) remain as implemented.  
Any future code change that alters CORE constraints requires updating this file and the fairness audit.
