from common import *

warnings.filterwarnings('ignore')
np.random.seed(42)


# ============================================================
# 4. BMOEA (Baseline)
# ============================================================

class BMOEA:
    def __init__(self, matrix, meal_tags, all_valid_idx, max_categories, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.max_categories = max_categories
        self.debug = debug

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS,
            crossover_rate=0.2, mutation_rate=0.9,
            min_foods=None, max_foods=None , has_meal_constraint=True,
            early_stop_feasible=None):

        if min_foods is None:
            min_foods = MIN_FOODS
        if max_foods is None:
            max_foods = MAX_FOODS

        n_valid = len(self.all_valid_idx)
        population = []
        for _ in range(pop_size):
            n_items = np.random.randint(min_foods, min(max_foods, n_valid) + 1)
            selected = np.random.choice(self.all_valid_idx, size=n_items, replace=False)
            population.append(selected)

        ffg = None
        ffg_evals = None
        feasible_found = False
        elite_size = 0
        n_evals = 0

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break

            scores, feasibles, _, n_foods_arr, contrib_diversities = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories, has_meal_constraint)
            n_evals += pop_size

            n_feas = feasibles.sum()
            if n_feas > 0 and not feasible_found:
                ffg = gen + 1
                ffg_evals = n_evals
                feasible_found = True
                if self.debug:
                    print(f"      [BMOEA] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}")

            if early_stop_feasible is not None and n_feas >= early_stop_feasible:
                if self.debug:
                    print(f"      [BMOEA] Gen {gen + 1}: Early stop, {n_feas} feasible")
                break

            feas_idx = np.where(feasibles)[0]
            num_valid = min(len(feas_idx), pop_size)
            if elite_size < num_valid:
                elite_size = num_valid
            new_pop = [population[i].copy() for i in feas_idx[:num_valid]]

            remaining = pop_size - len(new_pop)
            if remaining > 0:
                infeas_idx = np.where(~feasibles)[0]
                if len(infeas_idx) > 0:
                    infeas_scores = np.maximum(scores[infeas_idx], 0)
                    s = infeas_scores.sum()
                    if s > 1e-10:
                        probs = infeas_scores / s
                        chosen = np.random.choice(infeas_idx, size=remaining, p=probs, replace=True)
                    else:
                        chosen = np.random.choice(infeas_idx, size=remaining, replace=True)
                    for idx in chosen:
                        new_pop.append(population[idx].copy())
                else:
                    for i in range(remaining):
                        idx = feas_idx[i % len(feas_idx)]
                        new_pop.append(population[idx].copy())

            population = new_pop[:pop_size]

            for i in range(elite_size, pop_size - 1, 2):
                if np.random.random() < crossover_rate:
                    combined = np.unique(np.concatenate([population[i], population[i + 1]]))
                    n_target = np.random.randint(min_foods, min(max_foods, len(combined)) + 1)
                    population[i] = np.random.choice(combined, size=n_target, replace=False)

            for i in range(pop_size):
                if np.random.random() < mutation_rate:
                    indices = population[i]
                    action = np.random.randint(0, 3)
                    if action == 0 and len(indices) < max_foods:
                        new_idx = np.random.choice(self.all_valid_idx)
                        population[i] = np.unique(np.concatenate([indices, [new_idx]]))
                    elif action == 1 and len(indices) > min_foods:
                        pos = np.random.randint(0, len(indices))
                        population[i] = np.delete(indices, pos)
                    else:
                        pos = np.random.randint(0, len(indices))
                        new_idx = np.random.choice(self.all_valid_idx)
                        indices[pos] = new_idx
                        population[i] = np.unique(indices)

        final_scores, final_feasibles, final_diversities, final_n_foods, contrib_diversities = evaluate_batch(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories, has_meal_constraint)

        if final_feasibles.sum() > 0:
            avg_diversity = float(final_diversities[final_feasibles].mean())
            avg_n_foods = float(final_n_foods[final_feasibles].mean())
            avg_contrib = float(contrib_diversities[final_feasibles].mean())
        else:
            avg_diversity = 0.0
            avg_n_foods = 0.0
            avg_contrib = 0.0

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'ffg_evals': ffg_evals if ffg_evals else n_evals,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
            'diversity': avg_diversity,
            'avg_n_foods': avg_n_foods,
            'contrib': avg_contrib
        }]


# ============================================================
# 5. ACMO-Diet (Baseline)
# ============================================================

class ACMODiet:
    def __init__(self, matrix, meal_tags, all_valid_idx, max_categories, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.max_categories = max_categories
        self.debug = debug

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS,
            min_foods=None, max_foods=None, has_meal_constraint=True,
            early_stop_feasible=None):

        if min_foods is None:
            min_foods = MIN_FOODS
        if max_foods is None:
            max_foods = MAX_FOODS

        population = []
        for _ in range(pop_size):
            if np.random.random() < 0.7:
                ind = random_individual(self.all_valid_idx, self.meal_tags)
            else:
                n_items = np.random.randint(min_foods, min(max_foods, len(self.all_valid_idx)) + 1)
                ind = np.random.choice(self.all_valid_idx, size=n_items, replace=False)
            if ind is None:
                ind = np.random.choice(self.all_valid_idx, min_foods, replace=False)
            population.append(ind)

        ffg = None
        ffg_evals = None
        n_evals = 0

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break

            scores, feasibles, _, n_foods_arr, contrib_diversities = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories,has_meal_constraint)
            n_evals += pop_size

            n_feas = feasibles.sum()
            if n_feas > 0 and ffg is None:
                ffg = gen + 1
                ffg_evals = n_evals
                if self.debug:
                    print(f"      [ACMO] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}")

            if early_stop_feasible is not None and n_feas >= early_stop_feasible:
                if self.debug:
                    print(f"      [ACMO] Gen {gen + 1}: Early stop, {n_feas} feasible")
                break

            best_idx = np.argmax(scores)
            new_pop = [population[best_idx].copy()]

            while len(new_pop) < pop_size:
                tour1 = np.random.choice(len(population), 3, replace=False)
                parent1_idx = tour1[np.argmax(scores[tour1])]
                tour2 = np.random.choice(len(population), 3, replace=False)
                parent2_idx = tour2[np.argmax(scores[tour2])]
                child = population[parent1_idx].copy()

                if np.random.random() < 0.5 and len(child) > 0:
                    parent2 = population[parent2_idx]
                    if len(parent2) > 0:
                        n_replace = max(1, min(len(child) // 3, len(parent2)))
                        if n_replace <= len(child):
                            positions = np.random.choice(len(child), size=n_replace, replace=False)
                            replace_foods = np.random.choice(parent2, size=min(n_replace, len(parent2)), replace=True)
                            for pos, food in zip(positions, replace_foods[:len(positions)]):
                                child[pos] = food
                    child = np.unique(child)

                if np.random.random() < 0.3 and len(child) > 0:
                    action = np.random.randint(0, 3)
                    if action == 0 and len(child) < max_foods:
                        new_food = np.random.choice(self.all_valid_idx)
                        child = np.unique(np.concatenate([child, [new_food]]))
                    elif action == 1 and len(child) > min_foods:
                        remove_pos = np.random.randint(0, len(child))
                        child = np.delete(child, remove_pos)
                    else:
                        replace_pos = np.random.randint(0, len(child))
                        new_food = np.random.choice(self.all_valid_idx)
                        child[replace_pos] = new_food
                        child = np.unique(child)

                if len(child) < min_foods:
                    additional = np.random.choice(self.all_valid_idx, min_foods - len(child), replace=False)
                    child = np.unique(np.concatenate([child, additional]))
                if len(child) > max_foods:
                    remove_count = len(child) - max_foods
                    remove_indices = np.random.choice(len(child), size=remove_count, replace=False)
                    child = np.delete(child, remove_indices)

                new_pop.append(child)

            population = new_pop

        final_scores, final_feasibles, final_diversities, final_n_foods, contrib_diversities = evaluate_batch(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories,has_meal_constraint)

        if final_feasibles.sum() > 0:
            avg_diversity = float(final_diversities[final_feasibles].mean())
            avg_n_foods = float(final_n_foods[final_feasibles].mean())
            avg_contrib = float(contrib_diversities[final_feasibles].mean())
        else:
            avg_diversity = 0.0
            avg_n_foods = 0.0
            avg_contrib = 0.0

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'ffg_evals': ffg_evals if ffg_evals else n_evals,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
            'diversity': avg_diversity,
            'avg_n_foods': avg_n_foods,
            'contrib': avg_contrib
        }]


# ============================================================
# 6. Memetic Algorithm (MA)
# ============================================================

class MemeticAlgorithm:
    def __init__(self, matrix, meal_tags, all_valid_idx, max_categories, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.max_categories = max_categories
        self.debug = debug

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS,
            min_foods=None, max_foods=None,has_meal_constraint=True,
            early_stop_feasible=None):

        if min_foods is None:
            min_foods = MIN_FOODS
        if max_foods is None:
            max_foods = MAX_FOODS

        n_valid = len(self.all_valid_idx)
        population = []
        for _ in range(pop_size):
            n_items = np.random.randint(min_foods, min(max_foods, n_valid) + 1)
            selected = np.random.choice(self.all_valid_idx, size=n_items, replace=False)
            population.append(selected)

        ffg = None
        ffg_evals = None
        feasible_found = False
        n_evals = 0
        best_feasible = None
        best_fitness = 0.0
        target_center = (target_array[:, 0] + target_array[:, 1]) / 2.0
        all_valid_nutrients = self.matrix[self.all_valid_idx][:, col_idx]

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break

            scores, feasibles, diversities, n_foods_arr, contrib_diversities = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories,has_meal_constraint)
            n_evals += pop_size

            n_feas = feasibles.sum()
            if n_feas > 0:
                feas_scores = scores[feasibles]
                best_local_idx = np.argmax(feas_scores)
                best_local_score = feas_scores[best_local_idx]
                if best_local_score > best_fitness:
                    best_fitness = best_local_score
                    feas_indices = np.where(feasibles)[0]
                    best_feasible = population[feas_indices[best_local_idx]].copy()
                if not feasible_found:
                    ffg = gen + 1
                    ffg_evals = n_evals
                    feasible_found = True

            if early_stop_feasible is not None and n_feas >= early_stop_feasible:
                break

            if n_feas > 0 and gen % 3 == 0:
                n_ls = min(5, n_feas)
                for idx in np.where(feasibles)[0][:n_ls]:
                    improved = self._local_search(
                        population[idx], target_array, col_idx,
                        target_center, all_valid_nutrients, max_iter=10, min_foods=min_foods
                    )
                    fit, _, feas, _, _ = evaluate_single(
                        improved, self.matrix, target_array, col_idx,
                        self.meal_tags, self.max_categories,has_meal_constraint
                    )
                    n_evals += 1
                    if feas and fit > best_fitness:
                        best_fitness = fit
                        best_feasible = improved.copy()
                        population[idx] = improved
                        scores[idx] = fit
                        feasibles[idx] = True

            feas_idx = np.where(feasibles)[0]
            num_valid = min(len(feas_idx), pop_size)
            new_pop = [population[i].copy() for i in feas_idx[:num_valid]]
            elite_size = num_valid

            remaining = pop_size - len(new_pop)
            if remaining > 0:
                infeas_idx = np.where(~feasibles)[0]
                if len(infeas_idx) > 0:
                    infeas_scores = np.maximum(scores[infeas_idx], 0)
                    s = infeas_scores.sum()
                    if s > 1e-10:
                        probs = infeas_scores / s
                        chosen = np.random.choice(infeas_idx, size=remaining, p=probs, replace=True)
                    else:
                        chosen = np.random.choice(infeas_idx, size=remaining, replace=True)
                    for idx in chosen:
                        new_pop.append(population[idx].copy())
                else:
                    for i in range(remaining):
                        idx = feas_idx[i % len(feas_idx)]
                        new_pop.append(population[idx].copy())

            population = new_pop[:pop_size]

            for i in range(elite_size, pop_size - 1, 2):
                if np.random.random() < 0.3:
                    combined = np.unique(np.concatenate([population[i], population[i + 1]]))
                    n_target = np.random.randint(min_foods, min(max_foods, len(combined)) + 1)
                    population[i] = np.random.choice(combined, size=n_target, replace=False)

            for i in range(pop_size):
                if np.random.random() < 0.7:
                    indices = population[i]
                    if len(indices) == 0:
                        continue
                    action = np.random.randint(0, 3)
                    if action == 0 and len(indices) < max_foods:
                        if np.random.random() < 0.3 and len(indices) >= min_foods:
                            current_total = self.matrix[indices][:, col_idx].sum(axis=0)
                            gap = target_center - current_total
                            candidates = np.random.choice(self.all_valid_idx, size=min(10, n_valid), replace=False)
                            best_score = -float('inf')
                            best_cand = None
                            for cand in candidates:
                                new_total = current_total + self.matrix[cand][col_idx]
                                score = -np.sum((new_total - target_center) ** 2)
                                if score > best_score:
                                    best_score = score
                                    best_cand = cand
                            if best_cand is not None:
                                indices = np.concatenate([indices, [best_cand]])
                        else:
                            new_idx = np.random.choice(self.all_valid_idx)
                            indices = np.concatenate([indices, [new_idx]])
                        population[i] = np.unique(indices)

                    elif action == 1 and len(indices) > min_foods:
                        current_total = self.matrix[indices][:, col_idx].sum(axis=0)
                        best_score = float('inf')
                        worst_pos = 0
                        for j, food in enumerate(indices):
                            without = current_total - self.matrix[food][col_idx]
                            score = np.sum((without - target_center) ** 2)
                            if score < best_score:
                                best_score = score
                                worst_pos = j
                        population[i] = np.delete(indices, worst_pos)

                    else:
                        pos = np.random.randint(0, len(indices))
                        old_food = indices[pos]
                        if np.random.random() < 0.3 and len(indices) >= min_foods:
                            current_total = self.matrix[indices][:, col_idx].sum(axis=0)
                            without = current_total - self.matrix[old_food][col_idx]
                            candidates = np.random.choice(self.all_valid_idx, size=min(10, n_valid), replace=False)
                            best_score = float('inf')
                            best_cand = old_food
                            for cand in candidates:
                                new_total = without + self.matrix[cand][col_idx]
                                score = np.sum((new_total - target_center) ** 2)
                                if score < best_score:
                                    best_score = score
                                    best_cand = cand
                            indices[pos] = best_cand
                        else:
                            indices[pos] = np.random.choice(self.all_valid_idx)
                        population[i] = np.unique(indices)

        final_scores, final_feasibles, final_diversities, final_n_foods, contrib_diversities = evaluate_batch(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories,has_meal_constraint)
        n_evals += pop_size

        if final_feasibles.sum() == 0 and best_feasible is not None:
            population[0] = best_feasible
            final_scores, final_feasibles, final_diversities, final_n_foods, contrib_diversities = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories,has_meal_constraint)
        elif final_feasibles.sum() > 0:
            feas_indices = np.where(final_feasibles)[0]
            top_indices = feas_indices[np.argsort(final_scores[feas_indices])[-3:]]
            for idx in top_indices:
                improved = self._local_search(
                    population[idx], target_array, col_idx,
                    target_center, all_valid_nutrients, max_iter=20, min_foods=min_foods
                )
                fit, _, feas, _, _ = evaluate_single(
                    improved, self.matrix, target_array, col_idx,
                    self.meal_tags, self.max_categories,has_meal_constraint
                )
                n_evals += 1
                if feas and fit > final_scores[idx]:
                    population[idx] = improved
                    final_scores[idx] = fit

        if final_feasibles.sum() > 0:
            avg_diversity = float(final_diversities[final_feasibles].mean())
            avg_n_foods = float(final_n_foods[final_feasibles].mean())
            avg_contrib = float(contrib_diversities[final_feasibles].mean())
        else:
            avg_diversity = 0.0
            avg_n_foods = 0.0
            avg_contrib = 0.0

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'ffg_evals': ffg_evals if ffg_evals else n_evals,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
            'diversity': avg_diversity,
            'avg_n_foods': avg_n_foods,
            'contrib': avg_contrib
        }]

    def _local_search(self, individual, target_array, col_idx, target_center,
                      all_valid_nutrients, max_iter=10, min_foods=None):
        if min_foods is None:
            min_foods = MIN_FOODS

        individual = np.array(individual, dtype=np.int32)
        if len(individual) < min_foods:
            return individual

        best_ind = individual.copy()
        best_total = self.matrix[best_ind][:, col_idx].sum(axis=0)
        best_dist = np.sum((best_total - target_center) ** 2)

        n_candidates = min(50, len(self.all_valid_idx))
        candidates = np.random.choice(self.all_valid_idx, size=n_candidates, replace=False)
        candidate_nutrients = self.matrix[candidates][:, col_idx]

        improved = True
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            positions = np.random.permutation(len(best_ind))

            for pos in positions:
                old_food = best_ind[pos]
                old_nutrients = self.matrix[old_food][col_idx]
                without = best_total - old_nutrients
                new_totals = without + candidate_nutrients
                distances = np.sum((new_totals - target_center) ** 2, axis=1)
                best_cand_idx = np.argmin(distances)

                if distances[best_cand_idx] < best_dist:
                    best_ind[pos] = candidates[best_cand_idx]
                    best_total = without + candidate_nutrients[best_cand_idx]
                    best_dist = distances[best_cand_idx]
                    improved = True
                    lo, hi = target_array[:, 0], target_array[:, 1]
                    if np.all(best_total >= lo) and np.all(best_total <= hi):
                        return best_ind

        return best_ind


# ============================================================
# 7. pymoo 多目标算法适配器
# ============================================================

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.age import AGEMOEA
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.algorithms.moo.rvea import RVEA
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.algorithms.moo.ctaea import CTAEA
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.core.sampling import Sampling

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not available, multi-objective algorithms will be skipped")


class DietProblem(Problem):
    def __init__(self, matrix, target_array, col_idx, meal_tags, all_valid_idx,
                 max_categories, min_foods=None, max_foods=None, has_meal_constraint=True):  # 添加参数
        self.matrix = matrix
        self.target_array = target_array
        self.col_idx = col_idx
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.max_categories = max_categories
        self.min_foods = min_foods if min_foods is not None else MIN_FOODS
        self.max_foods = max_foods if max_foods is not None else MAX_FOODS
        self.has_meal_constraint = has_meal_constraint  # 保存

        n_var = self.max_foods
        xl = np.zeros(n_var) - 1
        xu = np.ones(n_var) * (len(all_valid_idx) - 1)
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        n_ind = x.shape[0]
        F = np.zeros((n_ind, 2))

        for i in range(n_ind):
            valid_mask = x[i] >= 0
            indices = self.all_valid_idx[x[i][valid_mask].astype(int)]

            if len(indices) < self.min_foods or len(indices) > self.max_foods:
                F[i, 0] = 1e6
                F[i, 1] = 0
                continue

            total = self.matrix[indices][:, self.col_idx].sum(axis=0)
            lo, hi = self.target_array[:, 0], self.target_array[:, 1]
            below = np.maximum(0, lo - total)
            above = np.maximum(0, total - hi)
            nutrient_penalty = float((below + above).sum())

            categories = set()
            for fid in indices:
                cat = get_food_category(fid)
                categories.add(cat)
            diversity_score = len(categories) / self.max_categories if self.max_categories > 0 else 0.0
            diversity_penalty = float((1 - diversity_score) * DIVERSITY_WEIGHT)

            F[i, 0] = nutrient_penalty + diversity_penalty

            # 三餐约束：根据 has_meal_constraint 控制
            if self.has_meal_constraint:
                meals = self.meal_tags[indices]
                meal_coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
            else:
                meal_coverage = 3  # 无约束时默认满分
            F[i, 1] = -meal_coverage

        out["F"] = F


class IntSampling(Sampling):
    def __init__(self, all_valid_idx, min_foods=None, max_foods=None):
        super().__init__()
        self.all_valid_idx = all_valid_idx
        self.min_foods = min_foods if min_foods is not None else MIN_FOODS
        self.max_foods = max_foods if max_foods is not None else MAX_FOODS

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), -1, dtype=int)
        for i in range(n_samples):
            n_items = np.random.randint(self.min_foods, min(self.max_foods, len(self.all_valid_idx)) + 1)
            selected = np.random.choice(len(self.all_valid_idx), size=n_items, replace=False)
            for j, idx in enumerate(selected):
                if j < problem.n_var:
                    X[i, j] = idx
        return X


def run_pymoo_algorithm(algo_name, matrix, target_array, col_idx, meal_tags, all_valid_idx,
                        max_categories, pop_size=2000, n_generations=50, debug=False,
                        min_foods=None, max_foods=None, has_meal_constraint=True,
                        seed=42, max_evals=None, early_stop_feasible=None):
    if min_foods is None:
        min_foods = MIN_FOODS
    if max_foods is None:
        max_foods = MAX_FOODS
    if max_evals is None:
        max_evals = MAX_EVALS

    default_result = {
        'ffg': n_generations + 1, 'ffg_evals': max_evals, 'fhr': 0.0, 'fsc': 0,
        'best_fitness': 0.0, 'n_evals': 0, 'diversity': 0.0,
        'avg_n_foods': 0.0, 'contrib': 0.0,
    }

    if not PYMOO_AVAILABLE:
        return default_result

    try:
        problem = DietProblem(matrix, target_array, col_idx, meal_tags, all_valid_idx,
                              max_categories, min_foods, max_foods, has_meal_constraint)
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        sampling = IntSampling(all_valid_idx, min_foods, max_foods)
        crossover = SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair())
        mutation = PM(prob=1.0, eta=20, vtype=float, repair=RoundingRepair())

        if algo_name == 'NSGA2':
            algorithm = NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation)
        elif algo_name == 'NSGA3':
            algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=sampling,
                              crossover=crossover, mutation=mutation)
        elif algo_name == 'MOEAD':
            algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=20, sampling=sampling,
                              crossover=crossover, mutation=mutation, prob_neighbor_mating=0.7)
        elif algo_name == 'RVEA':
            algorithm = RVEA(ref_dirs=ref_dirs, pop_size=pop_size, sampling=sampling,
                             crossover=crossover, mutation=mutation)
        elif algo_name == 'SPEA2':
            algorithm = SPEA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation)
        elif algo_name == 'AGEMOEA':
            algorithm = AGEMOEA(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation)
        elif algo_name == 'SMSEMOA':
            algorithm = SMSEMOA(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation)
        elif algo_name == 'CTAEA':
            algorithm = CTAEA(ref_dirs=ref_dirs, sampling=sampling, crossover=crossover, mutation=mutation)
        else:
            return default_result

        # Generations capped by eval budget
        max_gen_by_budget = max(1, int(np.ceil(max_evals / max(pop_size, 1))))
        n_generations = min(n_generations, max_gen_by_budget)
        algorithm.setup(problem, termination=('n_gen', n_generations), seed=int(seed))

        n_evals = 0
        first_feasible_gen = None
        first_feasible_evals = None
        best_fitness = 0.0
        all_feasible_individuals = []

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break
            algorithm.next()
            n_evals += pop_size

            if algorithm.pop is not None:
                feas_in_current_gen = 0
                for ind in algorithm.pop:
                    x = ind.X
                    valid_mask = x >= 0
                    indices = all_valid_idx[x[valid_mask].astype(int)]
                    if len(indices) >= min_foods:
                        f, _, fe, d, c = evaluate_single(indices, matrix, target_array, col_idx, meal_tags,
                                                         max_categories,has_meal_constraint)
                        f = min(1.0, max(0.0, f))
                        if fe and f > best_fitness:
                            best_fitness = f
                        if fe:
                            all_feasible_individuals.append((indices.copy(), f, d, len(indices), c))
                            feas_in_current_gen += 1
                            if first_feasible_gen is None:
                                first_feasible_gen = gen + 1
                                first_feasible_evals = n_evals
                                if debug:
                                    print(f"      [pymoo {algo_name}] Gen {gen + 1}: FIRST FEASIBLE!")
                if early_stop_feasible is not None and feas_in_current_gen >= early_stop_feasible:
                    break

        n_feas = len(all_feasible_individuals)
        if n_feas > 0:
            avg_diversity = np.mean([d for _, _, d, _, _ in all_feasible_individuals])
            avg_n_foods = np.mean([nf for _, _, _, nf, _ in all_feasible_individuals])
            avg_contrib = np.mean([cd for _, _, _, _, cd in all_feasible_individuals])
        else:
            avg_diversity = 0.0
            avg_n_foods = 0.0
            avg_contrib = 0.0

        return {
            'ffg': first_feasible_gen if first_feasible_gen else n_generations + 1,
            'ffg_evals': first_feasible_evals if first_feasible_evals else n_evals,
            'fhr': 1.0 if n_feas > 0 else 0.0,
            'fsc': n_feas,
            'best_fitness': best_fitness,
            'n_evals': n_evals,
            'diversity': avg_diversity,
            'avg_n_foods': avg_n_foods,
            'contrib': avg_contrib
        }

    except Exception as e:
        print(f"      [pymoo {algo_name}] Error: {e}")
        return default_result


def evaluate_single(individual, matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint=True):
    n_foods = len(individual)
    if n_foods < MIN_FOODS or n_foods > MAX_FOODS:
        return 0.0, None, False, 0.0, 0.0

    total = matrix[individual][:, col_idx].sum(axis=0)
    lo, hi = target_array[:, 0], target_array[:, 1]
    below = np.maximum(0, lo - total)
    above = np.maximum(0, total - hi)
    nutrient_penalty = float((below + above).sum())
    nutrient_feasible = nutrient_penalty < 1e-6

    if has_meal_constraint and meal_tags is not None:
        meals = meal_tags[individual]
        meal_coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
        meal_penalty = float((3 - meal_coverage) * MEAL_COVERAGE_WEIGHT)
        meal_feasible = (meal_coverage == 3)
    else:
        meal_penalty = 0.0
        meal_feasible = True

    categories = set()
    for fid in individual:
        categories.add(get_food_category(fid))
    diversity_score = len(categories) / max_categories if max_categories > 0 else 0.0
    diversity_penalty = float((1 - diversity_score) * DIVERSITY_WEIGHT)

    contrib_diversity = compute_contribution_diversity(individual, matrix, col_idx)

    total_penalty = nutrient_penalty + meal_penalty + diversity_penalty
    fitness = 1.0 / (1.0 + max(0.0, total_penalty))
    feasible = nutrient_feasible and meal_feasible

    return fitness, total, feasible, diversity_score, contrib_diversity