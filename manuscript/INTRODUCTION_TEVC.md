# TEVC Introduction Draft (placeholder citations [1]–[12])

> Synced into `manuscript/nutriopt_bench.tex` with `\cite{...}` keys.
> Placeholder numbers below map to the short reference list at the end.

## I. Introduction

Evolutionary multiobjective optimization (EMO) is routinely applied to constrained combinatorial problems whose decision variables are discrete and whose feasible sets may be sparse [1], [2]. *Range-Constrained Food Assembly* (RCFA) is one such problem: select a binary subset of candidate foods so that aggregate nutrient totals lie inside a *nutrient interval box*, optionally with structural coverage constraints. From an evolutionary computation standpoint, RCFA is constrained binary subset selection under coupled interval inequalities. Its value for EMO research is methodological—large decision spaces, controllable constraint geometry, and an unambiguous feasibility predicate—rather than any claim about dietary adequacy or clinical outcomes. All statements in this paper concern optimization behavior (e.g., Feasible Hit Rate, Friedman tests, and Kendall τ), not nutritional effectiveness.

Despite this clear search structure, algorithmic evaluation in nutrition-oriented combinatorial optimization has been shaped by three widely shared but weakly tested assumptions. First, enlarging the food library is presumed to make search harder, so that small recipe catalogs are treated as convenient proxies for realistic pools. Second, increasing the number of nutrient interval constraints is presumed to raise difficulty in a largely uniform manner. Third, algorithm rankings obtained on recipe-scale instances are presumed to generalize to large survey-derived catalogs such as NHANES. These assumptions are consequential: a substantial line of EMO and related menu-planning studies reports comparisons on comparatively small recipe datasets [3], [4], [5], [6]. If the assumptions fail, published orderings may characterize an easy regime rather than the intended combinatorial class.

Existing evaluation practice exhibits three corresponding defects. (a) *Missing controllable difficulty axes.* Food-library scale, interval sparsity of the nutrient box, and nutrient-constraint dimensionality are rarely varied as independent experimental factors. Without an orthogonal design, performance differences cannot be attributed to a specific structural source of hardness. (b) *Confounded failure modes.* Zero or near-zero success is often read as evolutionary search failure, yet an instance may simply have an empty feasible set. Conflating algorithmic failure with inherent infeasibility biases solver comparisons and obscures what a benchmark is measuring. (c) *Non-portable rankings.* Cross-regime ranking concordance is seldom quantified. When encodings, budgets, metrics, and instance families differ across papers, “algorithm A beats B” claims do not transfer [2], [7], [8]. A related protocol gap is the scarcity of fair evaluation-budget control and the mixing of heterogeneous success criteria, with insufficient separation between easy recipe regimes and NHANES-style real food pools [9], [10].

This paper responds with an *experimental survey* centered on NutriOpt-Bench: a fair-budget benchmark suite for RCFA together with a critical examination of how EMO solvers behave when difficulty is controlled. NutriOpt-Bench freezes a common binary RCFA formulation and independently varies three axes under matched resources (population size 500, 100,000 evaluations, 10 seeds). Campaign **E1** scans food-library scale from recipe catalogs through NHANES subsamples to the full pool of approximately 210,000 foods. Campaign **E2** varies interval sparsity via width factors {0.5, 1.0, 1.5}, inducing tight-to-relaxed nutrient interval boxes; this is the primary hardness axis in our design. Campaign **E3** varies nutrient dimensionality over {6, 9, 12, 16} simultaneous interval constraints. Campaign **E4** performs cross-regime meta-analysis: Friedman ranks within regimes and Kendall τ across regimes quantify ranking concordance among shared solvers, spanning memetic/feasibility-biased evolutionary algorithms, NSGA-II, indicator-style EAs, reference-vector methods, and greedy/random baselines [1], [2], [7], [8], [6].

PRIMARY findings can be stated briefly and without crowning a universal champion. Interval sparsity is the most salient hardness factor for RCFA under the tested protocol: mean Feasible Hit Rate (FHR) changes sharply across sparsity levels, whereas enlarging the NHANES catalog alone does *not* induce FHR collapse under a fixed 100,000-evaluation budget. Nutrient dimensionality remains near a performance ceiling in low-dimensional settings; a pronounced degradation appears mainly for weaker solvers at 16 constraints. Recipe instances form a trivial ceiling regime and must not be used to rank algorithms intended for NHANES-scale pools. Among eight shared algorithms, cross-regime Kendall τ remains high (τ = 0.909–0.982): absolute difficulty can move substantially while relative tiers stay largely stable, with memetic/feasibility-biased EAs and NSGA-II occupying a leading group and no wholesale rank reversal across hardness regimes. Exact calibration on reduced pools further shows that some targets are intrinsically infeasible; reporting conventions must therefore separate empty feasible sets from search failure. We do **not** claim a single best algorithm for all RCFA regimes.

The contributions of this experimental survey are as follows.

1. **(C1)** Formulate RCFA as interval-box constrained binary subset selection, and construct NutriOpt-Bench with three orthogonal difficulty axes: food-library scale, interval sparsity, and nutrient-constraint dimensionality.
2. **(C2)** Empirically demonstrate that interval sparsity dominates problem hardness, while food-catalog scale alone does not induce FHR collapse under a fixed evaluation budget; recipe benchmark regimes are trivial and non-generalizable to NHANES-style real food pools.
3. **(C3)** Quantitatively verify strong cross-regime ranking concordance of EMO solvers via Kendall τ: memetic/feasibility-biased evolutionary algorithms and NSGA-II form a stable top-performing family across hardness regimes, without asserting a universal champion.
4. **(C4)** Propose a methodological distinction between algorithm search failure and inherent instance infeasibility for constrained subset-selection benchmarking, with a controlled fair-budget protocol and fully reproducible artifacts.

The remainder of this paper is organized as follows. Section II defines RCFA and situates related work from an evolutionary-computation perspective. Section III introduces the NutriOpt-Bench protocol and fair-budget controls. Sections IV–V present the E1–E3 PRIMARY experiments on food-library scale, interval sparsity, and nutrient dimensionality. Section VI reports the E4 cross-regime meta-analysis. Section VII discusses implications for EA search mechanisms. Section VIII states limitations and future directions. Section IX concludes.

---

## Proposed contribution bullet points

- **C1.** Formulate RCFA as interval-box constrained binary subset selection, and construct NutriOpt-Bench with three orthogonal difficulty axes: food library scale, interval sparsity, nutrient constraint dimensionality.
- **C2.** Empirically demonstrate that interval sparsity dominates problem hardness, while food catalog scale alone does not induce FHR collapse under fixed evaluation budget; recipe benchmark regimes are trivial and non-generalizable to NHANES-style real food pools.
- **C3.** Quantitatively verify strong cross-regime ranking concordance of EMO solvers via Kendall τ: memetic/feasibility-biased evolutionary algorithms and NSGA2 form a stable top-performing family across hardness regimes.
- **C4.** Propose a methodological distinction between algorithm search failure and inherent instance infeasibility for constrained subset-selection benchmarking, with controlled fair-budget experimental protocol and fully reproducible artifacts.

---

## References (Introduction placeholders only)

| # | Placeholder role (to be wired to full DOI-verified bibliography) |
|---|------------------------------------------------------------------|
| [1] | NSGA-II / elitist MOEA baseline method |
| [2] | Decomposition / indicator / archive EMO baselines (e.g., MOEA/D, SPEA2 lineage) |
| [3] | Evolutionary weekly dietary menu planning |
| [4] | Computer-based dietary menu planning |
| [5] | Evolutionary dietary menu planning (conference lineage) |
| [6] | Constrained multiobjective diet recommendation (ACMO-Diet) |
| [7] | Reference-vector many-objective EA (RVEA) |
| [8] | Constrained two-archive EA (C-TAEA) |
| [9] | Classical LP subsistence diet model |
| [10] | Classical LP palatable-diet model |
| [11] | Fair-budget / benchmark comparison methodology (EMO) |
| [12] | Memetic / feasibility-oriented EA mechanism literature |

*Note:* Clinical nutrition, health-outcome, and biomedical relation-extraction references are intentionally out of scope for this Introduction.
