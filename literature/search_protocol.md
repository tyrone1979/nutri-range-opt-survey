# Systematic Search Protocol — NutriOpt-Bench Survey

## Purpose

Identify peer-reviewed and archival studies that formulate **diet / meal / nutrition planning as an optimization problem**, with emphasis on structural differences in representation, constraints, and evaluation—not merely “used a genetic algorithm.”

## Databases (planned primary sources)

1. IEEE Xplore  
2. ACM Digital Library  
3. Scopus  
4. Web of Science  
5. PubMed / MEDLINE  
6. ScienceDirect  
7. SpringerLink  
8. arXiv (cs.NE, math.OC, cs.AI) — labeled as preprint when included  

## Date range

- Primary window: **1990-01-01 → 2026-09-01**  
- Classic OR diet models (e.g., Stigler) may be cited historically outside this window.

## Search concept blocks

| Block | Example terms |
|-------|----------------|
| A Application | diet OR “meal plan*” OR “menu plan*” OR nutrition* OR “food selection” OR “dietary pattern” |
| B Optimization | optimi* OR “integer programming” OR MILP OR “constraint programming” OR “evolutionary” OR “genetic algorithm” OR NSGA OR “particle swarm” OR metaheuristic* |
| C Constraints | nutrient* OR calorie* OR “dietary reference” OR DRI OR “feasible region” OR interval OR range |

Queries are listed in `search_queries.md`.

## Screening stages

1. **Identification:** export records; remove exact DOI/title duplicates.  
2. **Title/abstract:** apply inclusion/exclusion (`inclusion_exclusion.md`).  
3. **Full text:** code variables into `literature_matrix.csv`.  
4. **Snowballing:** forward/backward citation chase from included seeds.  

## Coding protocol

Two independent coders preferred; disagreements resolved by discussion. Each row must have verifiable bibliographic fields (title, year, venue/DOI when available).

## PRISMA note

A PRISMA-style flow diagram will be finalized after the live database search is completed. The matrix shipped in this repository is a **seed corpus of verified landmark papers**, not the final screened set. Counts in the manuscript must be updated when the search is finished.

## Integrity rules

- Do **not** invent papers, DOIs, or years.  
- Preprints must be marked.  
- Algorithm-only papers with no diet/nutrition application are excluded unless used as method baselines (Deb NSGA-II, etc.) and cited as methodological references, not diet studies.
