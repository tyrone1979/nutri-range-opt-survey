# Search Queries (Draft)

Combine blocks with AND; OR within blocks. Adapt field codes per database.

## Core Boolean (Scopus-style)

```
TITLE-ABS-KEY(
  (diet OR "meal planning" OR "meal plan" OR "menu planning" OR "food selection" OR "nutrient optimization" OR "dietary optimization")
  AND
  ("integer programming" OR MILP OR "linear programming" OR "constraint programming" OR "genetic algorithm" OR "evolutionary algorithm" OR NSGA OR "particle swarm" OR "simulated annealing" OR metaheuristic OR "multi-objective")
)
```

## IEEE / ACM variants

- `"meal planning" AND ("genetic algorithm" OR NSGA-II OR "particle swarm")`  
- `"diet optimization" AND ("integer programming" OR evolutionary)`  
- `"nutrient constraints" AND (evolutionary OR metaheuristic)`  
- `"personalized nutrition" AND optimization AND (meal OR diet)`  

## PubMed

```
("Diet"[Mesh] OR meal planning OR menu planning) AND (Algorithms[Mesh] OR "integer programming" OR "genetic algorithm")
```

## Snowball seeds (verified landmarks)

Use the DOI/title list in `literature_matrix.csv` as seeds for citation chaining.
