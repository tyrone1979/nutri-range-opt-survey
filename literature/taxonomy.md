# Taxonomy of Optimization-Based Diet Planning

## Purpose

Classify studies by **orthogonal structural axes**, not by algorithm brand names. NutriOpt-Bench occupies a specific cell: **binary/index food-subset selection × interval nutrient targets × cardinality (+ optional meal coverage) × large discrete search spaces**.

## Axis 1 — Decision representation

| Code | Meaning |
|------|---------|
| CONT | Continuous portions / grams |
| INT_SERV | Integer servings |
| BIN | Binary food inclusion |
| MIXED | Continuous + discrete |
| HIER | Hierarchical meal → dish → ingredient |

## Axis 2 — Nutritional constraint geometry

| Code | Meaning |
|------|---------|
| POINT | Equality / single target value |
| LB | Lower bounds only |
| UB | Upper bounds only |
| INTERVAL | Box constraints \([L,U]\) |
| DEV | Weighted deviation from targets |
| STOCH | Probabilistic / robust / chance constraints |

## Axis 3 — Structural constraints

Food-count, meal coverage, category coverage, allergies, budget, preference, repetition, availability, cultural/religious rules, multi-day linking.

## Axis 4 — Optimization paradigm

Exact (LP/MILP/CP), SOEA, dominance-MOEA, decomposition-MOEA, indicator-MOEA, constraint-dominance, swarm, local search, hybrid, learning-assisted.

## Axis 5 — Evaluation paradigm

Objective value, feasibility rate, time-to-feasibility, feasible-set size, diversity, nutrient deviation, preference scores, runtime, robustness, scalability.

## Salient gaps (hypothesis to test empirically)

1. Many published diet optimizers evaluate on **small** food catalogs (\(<10^3\)) with **continuous** portions — a different regime from large binary/index assembly.  
2. **Interval** targets with **hard feasibility** are less often the primary reported metric than soft deviation scores.  
3. Public **reproducible benchmarks** with fixed seeds, budgets, and raw CSVs are rare.  
4. Algorithm rankings are often reported as universal; **regime dependence** (scale × sparsity × dimension) is under-studied.

These gap statements must be updated after completing the literature matrix coding and NutriOpt-Bench experiments; they must not be stated as proven facts without evidence.
