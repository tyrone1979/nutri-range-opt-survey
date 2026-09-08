# PROBLEM_FORMULATION.md — NutriOpt-Bench

> **TEVC Phase II:** The frozen general formulation is  
> **`docs/RCFA_FORMULATION.md`** (Range-Constrained Food Assembly).  
> This file remains as an implementation-matched companion; prefer RCFA naming in the TEVC manuscript.

**Ground truth:** `common.py`, `baseline_algorithms.py`, `e1_scaling.py`  
**Status:** Frozen for manuscript Section 4 (implementation-matched); superseded in naming by RCFA freeze.

---

## 1. Problem name

**Range-Constrained Diet Assembly (RCDA)** — discrete food-subset selection under nutritional box constraints.

## 2. Sets and data

- Food index set \(F = \{0,\ldots,n-1\}\) (rows of the nutritional matrix after preprocessing).
- Nutrient index set \(J = \{1,\ldots,m\}\) (active nutrient columns for a target).
- Matrix \(A \in \mathbb{R}^{n \times m_{\mathrm{full}}}\); for a target, columns `col_idx` select \(A_{:J}\).
- Meal tag \(\tau_i \in \{0,1,2\}\) (breakfast/lunch/dinner) when available; else meal constraints disabled.
- Target bounds \(L,U \in \mathbb{R}^{|J|}\).

## 3. Decision variables

Let \(\mathcal{S} \subseteq F\) be the selected multiset of foods after `unique` operators (implementation favors sets).

Cardinality:

\[
k_{\min} \le |\mathcal{S}| \le k_{\max}
\]

| Regime | \(k_{\min}\) | \(k_{\max}\) |
|--------|-------------:|-------------:|
| NHANES (E1 FoodSpace) | 5 | 30 |
| Recipe datasets | 3 | 5 |
| Global `common` defaults | 3 | 30 |

**Not modeled:** continuous gram decision variables. NHANES `grams` only define the nutrient content of each row instance.

Equivalent binary form (used by MILP baseline):

\[
x_i \in \{0,1\}, \quad
k_{\min} \le \sum_i x_i \le k_{\max}, \quad
\mathcal{S} = \{i : x_i = 1\}.
\]

## 4. Nutritional interval constraints (hard)

\[
L_j \le \sum_{i \in \mathcal{S}} A_{ij} \le U_j \quad \forall j \in J.
\]

Violation:

\[
V_{\mathrm{nut}}(\mathcal{S}) = \sum_{j\in J}\Big(\max(0,L_j-t_j)+\max(0,t_j-U_j)\Big),\quad t=A_{\mathcal{S},J}\mathbf{1}.
\]

Nutrient-feasible iff \(V_{\mathrm{nut}} < 10^{-6}\).

## 5. Meal coverage (hard when enabled)

\[
c_m = \mathbf{1}\{\exists i\in\mathcal{S}:\tau_i=m\}, \quad m\in\{0,1,2\},
\quad C=\sum_m c_m.
\]

Meal-feasible iff \(C=3\). Soft penalty when scoring: \((3-C)\cdot 100\).

## 6. Diversity (soft only)

Category score \(d_{\mathrm{cat}}\) and contribution score \(d_{\mathrm{contrib}}\) as in `common.py`. Soft penalties:

\[
P_{\mathrm{div}} = (1-d_{\mathrm{cat}})\cdot 10 + (1-d_{\mathrm{contrib}})\cdot 5.
\]

**Do not** treat diversity as a feasibility requirement.

## 7. Feasibility

\[
\mathrm{Feasible}(\mathcal{S}) \iff
\big(k_{\min}\le|\mathcal{S}|\le k_{\max}\big)
\land (V_{\mathrm{nut}} < 10^{-6})
\land \big(\neg\mathrm{meal\_flag} \lor C=3\big).
\]

## 8. Scalar fitness (native EAs)

\[
P = V_{\mathrm{nut}} + P_{\mathrm{meal}} + P_{\mathrm{div}},
\quad
f(\mathcal{S}) = \frac{1}{1+\max(0,P)}.
\]

Maximize \(f\). Perfect nutrient+meal feasibility with zero soft penalties yields \(f=1\) only if diversity penalties also vanish (rare); **BestFit=1 is not required for hard feasibility**.

## 9. Bi-objective surrogate (pymoo path)

\[
\min \; f_1 = V_{\mathrm{nut}} + (1-d_{\mathrm{cat}})\cdot 10,
\quad
\min \; f_2 = -C
\]

(with \(C:=3\) if meal constraints off). No explicit constraint functions (`n_constr=0`).

## 10. Distinction from related problems

| Problem | NutriOpt-Bench? |
|---------|-----------------|
| Continuous LP diet (Stigler-type) | No |
| Serving-size MILP with recipes | Related, not identical |
| Food recommender ranking | No |
| Calorie-only tracking | No |
| RCDA binary/index assembly under intervals | **Yes** |

## 11. Intrinsically infeasible targets

If no \(x\in\{0,1\}^n\) satisfies nutritional (+ meal) constraints under cardinality, the instance is **infeasible**. Algorithm failure must be distinguished from instance infeasibility (MILP / bounding pre-check).
