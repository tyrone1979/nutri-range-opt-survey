# Inclusion and Exclusion Criteria

## Inclusion (must satisfy all)

1. The work addresses **diet, meal, menu, or food selection / nutrition planning**.  
2. There is an explicit **optimization, constraint satisfaction, or search** formulation (exact, heuristic, metaheuristic, or hybrid).  
3. Nutritional requirements appear as **objectives and/or constraints** (point, bound, or interval).  
4. Peer-reviewed article, refereed conference paper, or archived preprint with stable identifier.  
5. Sufficient detail to code at least: decision type, nutrient constraint type, optimization paradigm.

## Exclusion

1. Pure food **recommendation / ranking** without nutrient hard or soft constraints.  
2. Calorie tracking apps / UI papers with no solver.  
3. Clinical nutrition RCTs without computational optimization.  
4. Generic EA tutorials with a toy diet example lacking formulation detail (may be cited narratively only).  
5. Duplicate reports of the same study (keep most complete version).  
6. Non-English full texts when an English version exists; otherwise note language limitation.

## Borderline cases

| Case | Decision |
|------|----------|
| Portion (continuous grams) diet LP/MILP | **Include** — distinct axis from binary assembly |
| Recipe recommendation with soft nutrition score | Include only if optimization/search is central |
| Multi-day meal planning | Include; code temporal structure under structural constraints |
| Sustainability / cost + nutrition | Include; code extra structural/preference constraints |
