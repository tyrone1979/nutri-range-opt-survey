
import warnings
import numpy as np
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
                crossover_rate=0.2, mutation_rate=0.9):

            n_valid = len(self.all_valid_idx)
            population = []
            for _ in range(pop_size):
                n_items = np.random.randint(MIN_FOODS, min(MAX_FOODS, n_valid) + 1)
                selected = np.random.choice(self.all_valid_idx, size=n_items, replace=False)
                population.append(selected)

            ffg = None
            feasible_found = False
            elite_size = 0
            n_evals = 0

            for gen in range(n_generations):
                if n_evals >= max_evals:
                    break

                scores, feasibles, _, n_foods_arr = evaluate_batch(
                    population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
                n_evals += pop_size

                n_feas = feasibles.sum()
                if n_feas > 0 and not feasible_found:
                    ffg = gen + 1
                    feasible_found = True
                    if self.debug:
                        print(f"      [BMOEA] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}")

                if n_feas >= 20:
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
                        n_target = np.random.randint(MIN_FOODS, min(MAX_FOODS, len(combined)) + 1)
                        population[i] = np.random.choice(combined, size=n_target, replace=False)

                for i in range(pop_size):
                    if np.random.random() < mutation_rate:
                        indices = population[i]
                        action = np.random.randint(0, 3)
                        if action == 0 and len(indices) < MAX_FOODS:
                            new_idx = np.random.choice(self.all_valid_idx)
                            population[i] = np.unique(np.concatenate([indices, [new_idx]]))
                        elif action == 1 and len(indices) > MIN_FOODS:
                            pos = np.random.randint(0, len(indices))
                            population[i] = np.delete(indices, pos)
                        else:
                            pos = np.random.randint(0, len(indices))
                            new_idx = np.random.choice(self.all_valid_idx)
                            indices[pos] = new_idx
                            population[i] = np.unique(indices)

            final_scores, final_feasibles, final_diversities, final_n_foods = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)

            if final_feasibles.sum() > 0:
                avg_diversity = float(final_diversities[final_feasibles].mean())
                avg_n_foods = float(final_n_foods[final_feasibles].mean())
            else:
                avg_diversity = 0.0
                avg_n_foods = 0.0

            return [{
                'ffg': ffg if ffg else n_generations + 1,
                'fhr': 1.0 if np.any(final_feasibles) else 0.0,
                'fsc': int(final_feasibles.sum()),
                'best_fitness': float(final_scores.max()),
                'n_evals': n_evals,
                'diversity': avg_diversity,
                'avg_n_foods': avg_n_foods,
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
                n_generations=N_GENERATIONS, max_evals=MAX_EVALS):

            population = []
            for _ in range(pop_size):
                if np.random.random() < 0.7:
                    ind = random_individual(self.all_valid_idx, self.meal_tags)
                else:
                    n_items = np.random.randint(MIN_FOODS, min(MAX_FOODS, len(self.all_valid_idx)) + 1)
                    ind = np.random.choice(self.all_valid_idx, size=n_items, replace=False)
                if ind is None:
                    ind = np.random.choice(self.all_valid_idx, MIN_FOODS, replace=False)
                population.append(ind)

            ffg = None
            n_evals = 0

            for gen in range(n_generations):
                if n_evals >= max_evals:
                    break

                scores, feasibles, _, n_foods_arr = evaluate_batch(
                    population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
                n_evals += pop_size

                n_feas = feasibles.sum()
                if n_feas > 0 and ffg is None:
                    ffg = gen + 1
                    if self.debug:
                        print(f"      [ACMO] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}")

                if n_feas >= 20:
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
                                replace_foods = np.random.choice(parent2, size=min(n_replace, len(parent2)),
                                                                 replace=True)
                                for pos, food in zip(positions, replace_foods[:len(positions)]):
                                    child[pos] = food
                        child = np.unique(child)

                    if np.random.random() < 0.3 and len(child) > 0:
                        action = np.random.randint(0, 3)
                        if action == 0 and len(child) < MAX_FOODS:
                            new_food = np.random.choice(self.all_valid_idx)
                            child = np.unique(np.concatenate([child, [new_food]]))
                        elif action == 1 and len(child) > MIN_FOODS:
                            remove_pos = np.random.randint(0, len(child))
                            child = np.delete(child, remove_pos)
                        else:
                            replace_pos = np.random.randint(0, len(child))
                            new_food = np.random.choice(self.all_valid_idx)
                            child[replace_pos] = new_food
                            child = np.unique(child)

                    if len(child) < MIN_FOODS:
                        additional = np.random.choice(self.all_valid_idx, MIN_FOODS - len(child), replace=False)
                        child = np.unique(np.concatenate([child, additional]))
                    if len(child) > MAX_FOODS:
                        remove_count = len(child) - MAX_FOODS
                        remove_indices = np.random.choice(len(child), size=remove_count, replace=False)
                        child = np.delete(child, remove_indices)

                    new_pop.append(child)

                population = new_pop

            final_scores, final_feasibles, final_diversities, final_n_foods = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)

            if final_feasibles.sum() > 0:
                avg_diversity = float(final_diversities[final_feasibles].mean())
                avg_n_foods = float(final_n_foods[final_feasibles].mean())
            else:
                avg_diversity = 0.0
                avg_n_foods = 0.0

            return [{
                'ffg': ffg if ffg else n_generations + 1,
                'fhr': 1.0 if np.any(final_feasibles) else 0.0,
                'fsc': int(final_feasibles.sum()),
                'best_fitness': float(final_scores.max()),
                'n_evals': n_evals,
                'diversity': avg_diversity,
                'avg_n_foods': avg_n_foods,
            }]


# ============================================================
# 6. pymoo 多目标算法适配器
# ============================================================

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.rvea import RVEA
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
    def __init__(self, matrix, target_array, col_idx, meal_tags, all_valid_idx, max_categories):
        self.matrix = matrix
        self.target_array = target_array
        self.col_idx = col_idx
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.max_categories = max_categories

        n_var = MAX_FOODS
        xl = np.zeros(n_var) - 1
        xu = np.ones(n_var) * (len(all_valid_idx) - 1)
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        n_ind = x.shape[0]
        F = np.zeros((n_ind, 2))

        for i in range(n_ind):
            valid_mask = x[i] >= 0
            indices = self.all_valid_idx[x[i][valid_mask].astype(int)]

            if len(indices) < MIN_FOODS or len(indices) > MAX_FOODS:
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

            meals = self.meal_tags[indices]
            meal_coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
            F[i, 1] = -meal_coverage

        out["F"] = F


class IntSampling(Sampling):
    def __init__(self, all_valid_idx):
        super().__init__()
        self.all_valid_idx = all_valid_idx

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), -1, dtype=int)
        for i in range(n_samples):
            n_items = np.random.randint(MIN_FOODS, min(MAX_FOODS, len(self.all_valid_idx)) + 1)
            selected = np.random.choice(len(self.all_valid_idx), size=n_items, replace=False)
            for j, idx in enumerate(selected):
                if j < problem.n_var:
                    X[i, j] = idx
        return X


def evaluate_single(individual, matrix, target_array, col_idx, meal_tags, max_categories):
    n_foods = len(individual)
    if n_foods < MIN_FOODS or n_foods > MAX_FOODS:
        return 0.0, None, False, 0.0

    total = matrix[individual][:, col_idx].sum(axis=0)
    lo, hi = target_array[:, 0], target_array[:, 1]
    below = np.maximum(0, lo - total)
    above = np.maximum(0, total - hi)
    nutrient_penalty = float((below + above).sum())
    nutrient_feasible = nutrient_penalty < 1e-6

    meals = meal_tags[individual]
    meal_coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
    meal_penalty = float((3 - meal_coverage) * MEAL_COVERAGE_WEIGHT)

    categories = set()
    for fid in individual:
        categories.add(get_food_category(fid))
    diversity_score = len(categories) / max_categories if max_categories > 0 else 0.0
    diversity_penalty = float((1 - diversity_score) * DIVERSITY_WEIGHT)

    total_penalty = nutrient_penalty + meal_penalty + diversity_penalty
    fitness = 1.0 / (1.0 + max(0.0, total_penalty))
    feasible = nutrient_feasible and (meal_coverage == 3)

    return fitness, total, feasible, diversity_score

# run_pymoo_algorithm
def run_pymoo_algorithm(algo_name, matrix, target_array, col_idx, meal_tags, all_valid_idx,
                        max_categories, pop_size=2000, n_generations=50, debug=False):
    default_result = {
        'ffg': n_generations + 1,
        'fhr': 0.0,
        'fsc': 0,
        'best_fitness': 0.0,
        'n_evals': 0,
        'diversity': 0.0,
        'avg_n_foods': 0.0,
    }

    if not PYMOO_AVAILABLE:
        return default_result

    try:
        problem = DietProblem(matrix, target_array, col_idx, meal_tags, all_valid_idx, max_categories)
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        sampling = IntSampling(all_valid_idx)
        crossover = SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair())
        mutation = PM(prob=1.0, eta=20, vtype=float, repair=RoundingRepair())

        if algo_name == 'NSGA2':
            algorithm = NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation)
        elif algo_name == 'NSGA3':
            algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=sampling,
                              crossover=crossover, mutation=mutation)
        elif algo_name == 'MOEAD':
            algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=15, sampling=sampling,
                              crossover=crossover, mutation=mutation)
        elif algo_name == 'RVEA':
            algorithm = RVEA(ref_dirs=ref_dirs, pop_size=pop_size, sampling=sampling,
                             crossover=crossover, mutation=mutation)
        else:
            return default_result

        algorithm.setup(problem, termination=('n_gen', n_generations), seed=42)

        n_evals = 0
        first_feasible_gen = None
        best_fitness = 0.0
        all_feasible_individuals = []

        for gen in range(n_generations):
            algorithm.next()
            n_evals += pop_size

            if algorithm.pop is not None:
                feas_in_current_gen = 0
                for ind in algorithm.pop:
                    x = ind.X
                    valid_mask = x >= 0
                    indices = all_valid_idx[x[valid_mask].astype(int)]
                    if len(indices) >= MIN_FOODS:
                        f, _, fe, d = evaluate_single(indices, matrix, target_array, col_idx, meal_tags, max_categories)
                        f = min(1.0, max(0.0, f))

                        if fe and f > best_fitness:
                            best_fitness = f

                        if fe:
                            all_feasible_individuals.append((indices.copy(), f, d, len(indices)))
                            feas_in_current_gen += 1
                            if first_feasible_gen is None:
                                first_feasible_gen = gen + 1
                                if debug:
                                    print(f"      [pymoo {algo_name}] Gen {gen + 1}: FIRST FEASIBLE!")

                if feas_in_current_gen >= 20:
                    if debug:
                        print(f"      [pymoo {algo_name}] Gen {gen + 1}: Early stop, {feas_in_current_gen} feasible")
                    break

        n_feas = len(all_feasible_individuals)
        if n_feas > 0:
            avg_diversity = np.mean([d for _, _, d, _ in all_feasible_individuals])
            avg_n_foods = np.mean([nf for _, _, _, nf in all_feasible_individuals])
        else:
            avg_diversity = 0.0
            avg_n_foods = 0.0

        return {
            'ffg': first_feasible_gen if first_feasible_gen else n_generations + 1,
            'fhr': 1.0 if n_feas > 0 else 0.0,
            'fsc': n_feas,
            'best_fitness': best_fitness,
            'n_evals': n_evals,
            'diversity': avg_diversity,
            'avg_n_foods': avg_n_foods,
        }

    except Exception as e:
        print(f"      [pymoo {algo_name}] Error: {e}")
        return default_result