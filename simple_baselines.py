"""
Simple baselines for NutriOpt-Bench: Random, Greedy, Random+Repair, Local Search.
"""
from __future__ import annotations

import numpy as np
from common import (
    evaluate_batch,
    compute_max_categories,
    random_individual,
    MEAL_COVERAGE_WEIGHT,
    DIVERSITY_WEIGHT,
)


def _decode_fitness_components(ind, matrix, target_array, col_idx, meal_tags, has_meal_constraint):
    total = matrix[ind][:, col_idx].sum(axis=0)
    lo, hi = target_array[:, 0], target_array[:, 1]
    below = np.maximum(0, lo - total)
    above = np.maximum(0, total - hi)
    nutrient_penalty = float((below + above).sum())
    meal_penalty = 0.0
    if has_meal_constraint:
        meals = meal_tags[ind]
        coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
        meal_penalty = float((3 - coverage) * MEAL_COVERAGE_WEIGHT)
    return nutrient_penalty + meal_penalty, nutrient_penalty < 1e-6 and (
        (not has_meal_constraint) or meal_penalty == 0.0
    )


def _pack_result(ffg, ffg_evals, fhr, fsc, best_fit, n_evals, diversity, n_foods, contrib, max_gen):
    return {
        "ffg": ffg if ffg is not None else max_gen + 1,
        "ffg_evals": ffg_evals if ffg_evals is not None else n_evals,
        "fhr": float(fhr),
        "fsc": int(fsc),
        "best_fitness": float(best_fit),
        "n_evals": int(n_evals),
        "diversity": float(diversity),
        "avg_n_foods": float(n_foods),
        "contrib": float(contrib),
    }


def _finalize(population, matrix, target_array, col_idx, meal_tags, max_categories,
              has_meal_constraint, ffg, ffg_evals, n_evals, max_gen):
    scores, feas, divs, nfoods, contribs = evaluate_batch(
        population, matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint
    )
    if feas.sum() > 0:
        diversity = float(divs[feas].mean())
        n_foods = float(nfoods[feas].mean())
        contrib = float(contribs[feas].mean())
    else:
        diversity = n_foods = contrib = 0.0
    return _pack_result(
        ffg, ffg_evals, 1.0 if feas.any() else 0.0, int(feas.sum()),
        float(scores.max()) if len(scores) else 0.0, n_evals, diversity, n_foods, contrib, max_gen
    )


def _rand_ind(all_valid_idx, meal_tags, min_foods, max_foods, has_meal_constraint):
    if has_meal_constraint:
        ind = random_individual(all_valid_idx, meal_tags)
        if ind is not None and min_foods <= len(ind) <= max_foods:
            return ind
    n = np.random.randint(min_foods, min(max_foods, len(all_valid_idx)) + 1)
    return np.random.choice(all_valid_idx, size=n, replace=False).astype(np.int32)


class RandomSearch:
    def run(self, matrix, meal_tags, all_valid_idx, target_array, col_idx,
            pop_size=500, max_evaluations=100000, n_generations=10000,
            min_foods=3, max_foods=30, has_meal_constraint=True, early_stop_feasible=None,
            seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        max_categories = compute_max_categories(all_valid_idx)
        best_pop = []
        ffg = ffg_evals = None
        n_evals = 0
        gen = 0
        while n_evals < max_evaluations and gen < n_generations:
            gen += 1
            batch = [
                _rand_ind(all_valid_idx, meal_tags, min_foods, max_foods, has_meal_constraint)
                for _ in range(min(pop_size, max_evaluations - n_evals))
            ]
            scores, feas, _, _, _ = evaluate_batch(
                batch, matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint
            )
            n_evals += len(batch)
            if feas.any() and ffg is None:
                ffg, ffg_evals = gen, n_evals
            # keep best pop_size by score
            combined = best_pop + batch
            cs, cf, _, _, _ = evaluate_batch(
                combined, matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint
            )
            order = np.argsort(-cs)[:pop_size]
            best_pop = [combined[i] for i in order]
            if early_stop_feasible and cf.sum() >= early_stop_feasible:
                break
        if not best_pop:
            best_pop = batch
        return [_finalize(best_pop, matrix, target_array, col_idx, meal_tags, max_categories,
                          has_meal_constraint, ffg, ffg_evals, n_evals, gen)]


class GreedyConstruct:
    """Greedy: start from meal seeds (if any), repeatedly add food minimizing nutrient violation."""

    def run(self, matrix, meal_tags, all_valid_idx, target_array, col_idx,
            pop_size=500, max_evaluations=100000, n_generations=10000,
            min_foods=3, max_foods=30, has_meal_constraint=True, early_stop_feasible=None,
            seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        max_categories = compute_max_categories(all_valid_idx)
        lo, hi = target_array[:, 0], target_array[:, 1]
        population = []
        ffg = ffg_evals = None
        n_evals = 0
        gen = 0

        def violation(ind):
            t = matrix[ind][:, col_idx].sum(axis=0)
            return float((np.maximum(0, lo - t) + np.maximum(0, t - hi)).sum())

        while n_evals < max_evaluations and len(population) < pop_size and gen < n_generations:
            gen += 1
            chosen = []
            if has_meal_constraint:
                for m in (0, 1, 2):
                    pool = all_valid_idx[meal_tags[all_valid_idx] == m]
                    if len(pool) == 0:
                        pool = all_valid_idx
                    chosen.append(int(np.random.choice(pool)))
            else:
                chosen = list(np.random.choice(all_valid_idx, size=min_foods, replace=False))

            ind = np.unique(np.array(chosen, dtype=np.int32))
            n_evals += 1
            while len(ind) < max_foods and n_evals < max_evaluations:
                # sample candidate pool
                cand = np.random.choice(all_valid_idx, size=min(50, len(all_valid_idx)), replace=False)
                best_v, best_c = None, None
                base_v = violation(ind)
                for c in cand:
                    if c in ind:
                        continue
                    trial = np.unique(np.concatenate([ind, [c]]))
                    v = violation(trial)
                    n_evals += 1
                    if best_v is None or v < best_v:
                        best_v, best_c = v, c
                    if n_evals >= max_evaluations:
                        break
                if best_c is None or best_v >= base_v and len(ind) >= min_foods:
                    break
                ind = np.unique(np.concatenate([ind, [best_c]]))
                if best_v < 1e-6 and len(ind) >= min_foods:
                    break

            population.append(ind)
            scores, feas, _, _, _ = evaluate_batch(
                [ind], matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint
            )
            if feas.any() and ffg is None:
                ffg, ffg_evals = gen, n_evals

        while len(population) < pop_size:
            population.append(_rand_ind(all_valid_idx, meal_tags, min_foods, max_foods, has_meal_constraint))
            n_evals += 1

        return [_finalize(population[:pop_size], matrix, target_array, col_idx, meal_tags, max_categories,
                          has_meal_constraint, ffg, ffg_evals, n_evals, gen)]


class RandomRepair:
    """
    Random individuals + repair (delete upper violators / add for deficits).

    Continues sampling+repair until the shared max_evaluations budget is
    exhausted (does NOT stop once pop_size individuals are built — that
    previously caused large budget under-use vs other algorithms).
    """

    def run(self, matrix, meal_tags, all_valid_idx, target_array, col_idx,
            pop_size=500, max_evaluations=100000, n_generations=10000,
            min_foods=3, max_foods=30, has_meal_constraint=True, early_stop_feasible=None,
            seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        max_categories = compute_max_categories(all_valid_idx)
        lo, hi = target_array[:, 0], target_array[:, 1]
        best_pop = []
        best_scores = []
        best_feas = []
        ffg = ffg_evals = None
        n_evals = 0
        gen = 0
        max_repair_steps = 20

        def _archive(ind, score, is_feas):
            if len(best_pop) < pop_size:
                best_pop.append(ind)
                best_scores.append(float(score))
                best_feas.append(bool(is_feas))
            else:
                worst = int(np.argmin(best_scores))
                if score > best_scores[worst]:
                    best_pop[worst] = ind
                    best_scores[worst] = float(score)
                    best_feas[worst] = bool(is_feas)

        while n_evals < max_evaluations and gen < n_generations:
            gen += 1
            ind = _rand_ind(all_valid_idx, meal_tags, min_foods, max_foods, has_meal_constraint)
            for _ in range(max_repair_steps):
                if n_evals >= max_evaluations:
                    break
                t = matrix[ind][:, col_idx].sum(axis=0)
                below = lo - t
                above = t - hi
                n_evals += 1
                if np.all(below <= 0) and np.all(above <= 0):
                    break
                if np.any(above > 0) and len(ind) > min_foods:
                    contrib = matrix[ind][:, col_idx]
                    sc = (contrib * np.maximum(above, 0)).sum(axis=1)
                    ind = np.delete(ind, int(np.argmax(sc)))
                elif np.any(below > 0) and len(ind) < max_foods:
                    need = np.maximum(below, 0)
                    cand = np.random.choice(all_valid_idx, size=min(30, len(all_valid_idx)), replace=False)
                    gains = (matrix[cand][:, col_idx] * need).sum(axis=1)
                    ind = np.unique(np.concatenate([ind, [cand[int(np.argmax(gains))]]]))
                else:
                    break

            if n_evals >= max_evaluations:
                if not best_pop:
                    best_pop = [ind]
                    best_scores = [0.0]
                break

            scores, feas, _, _, _ = evaluate_batch(
                [ind], matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint
            )
            n_evals += 1
            if feas.any() and ffg is None:
                ffg, ffg_evals = gen, n_evals
            _archive(ind, float(scores[0]), bool(feas[0]))

            if early_stop_feasible is not None and sum(best_feas) >= early_stop_feasible:
                break

        if not best_pop:
            best_pop = [_rand_ind(all_valid_idx, meal_tags, min_foods, max_foods, has_meal_constraint)]

        return [_finalize(best_pop[:pop_size], matrix, target_array, col_idx, meal_tags, max_categories,
                          has_meal_constraint, ffg, ffg_evals, n_evals, gen)]
