"""
NutriSwap-GA: 超越 RVEA 的优化版本
核心改进:
  1. 目标向量引导替换: 按营养素缺失方向选择食物，而非最近邻
  2. 双档案库: 可行解档案 + 多样性档案，交叉时注入历史精英
  3. 自适应变异: 前期高探索(随机变异)，后期高利用(向量引导)
  4. 多样性交叉: 合并后按类别去重保留
  5. 预计算梯度矩阵: 每个目标只算一次，O(1)查询替换
"""

from baseline_algorithms import *
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 3. NutriSwap-GA
# ============================================================

class NutriSwapGA:
    """
    NutriSwap-GA v7: 强制食物个数 15-30 + 多样性约束
    """

    def __init__(self, matrix, meal_tags, all_valid_idx, max_categories, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = np.array(all_valid_idx, dtype=np.int32)
        self.all_valid_idx_set = set(int(fid) for fid in all_valid_idx)
        self.max_categories = max_categories
        self.debug = debug

        self.food_to_pos = None
        self.pos_to_food = None
        self.meal_positions = None
        self.food_categories = None
        self.category_to_positions = None
        self.category_list = None
        self.category_pos_arrays = None

    def _precompute_structure(self):
        self.pos_to_food = self.all_valid_idx
        self.food_to_pos = {int(fid): i for i, fid in enumerate(self.all_valid_idx)}

        valid_meal_tags = self.meal_tags[self.all_valid_idx]
        self.meal_positions = {
            0: np.where(valid_meal_tags == 0)[0],
            1: np.where(valid_meal_tags == 1)[0],
            2: np.where(valid_meal_tags == 2)[0],
        }

        self.food_categories = np.array([
            str(int(fid))[:2] if len(str(int(fid))) >= 2 else str(int(fid))
            for fid in self.all_valid_idx
        ])
        self.category_to_positions = {}
        for i, cat in enumerate(self.food_categories):
            if cat not in self.category_to_positions:
                self.category_to_positions[cat] = []
            self.category_to_positions[cat].append(i)

        self.category_list = list(self.category_to_positions.keys())
        self.category_pos_arrays = [np.array(v, dtype=np.int32) for v in self.category_to_positions.values()]

    def _compute_dynamic_gradient(self, target_array, col_idx, population=None):
        target_center = (target_array[:, 0] + target_array[:, 1]) / 2.0
        valid_matrix = self.matrix[self.all_valid_idx][:, col_idx]

        if population is not None and len(population) > 0:
            avg_total = np.zeros(len(col_idx), dtype=np.float32)
            count = 0
            for ind in population:
                ind_arr = np.array(ind, dtype=np.int32)
                if len(ind_arr) >= MIN_FOODS:
                    avg_total += self.matrix[ind_arr][:, col_idx].sum(axis=0)
                    count += 1
            if count > 0:
                avg_total /= count
                gaps = np.maximum(0, target_center - avg_total)
                weights = 1.0 + gaps / (target_center + 1e-6)
            else:
                weights = np.ones(len(col_idx), dtype=np.float32)
        else:
            weights = np.ones(len(col_idx), dtype=np.float32)

        diff = valid_matrix - target_center
        weighted_diff = diff * weights
        distances = np.sum(weighted_diff ** 2, axis=1)
        max_dist = distances.max() if distances.max() > 0 else 1.0
        gradient = 1.0 - (distances / max_dist)
        return gradient

    def _vector_guided_replace(self, food_id, current_individual, target_array, col_idx,
                               gradient, top_k=30):
        food_id = int(food_id)
        current_individual = np.array(current_individual, dtype=np.int32)

        food_pos = self.food_to_pos.get(food_id)
        if food_pos is None:
            return food_id

        current_total = self.matrix[current_individual][:, col_idx].sum(axis=0)
        current_without = current_total - self.matrix[food_id][col_idx]
        target_center = (target_array[:, 0] + target_array[:, 1]) / 2.0

        current_cats = set(get_food_category(fid) for fid in current_individual)
        current_meals = set(self.meal_tags[current_individual])
        missing_meals = {0, 1, 2} - current_meals

        combined = gradient.copy()
        for cat, positions in self.category_to_positions.items():
            if cat not in current_cats:
                combined[np.array(positions, dtype=np.int32)] += 0.5
        for meal in missing_meals:
            combined[self.meal_positions[meal]] += 0.6

        combined[food_pos] = -999
        candidate_positions = np.argpartition(combined, -top_k)[-top_k:]
        candidate_positions = candidate_positions[np.argsort(combined[candidate_positions])[::-1]]

        best_food = food_id
        best_score = -float('inf')

        for pos in candidate_positions:
            if combined[pos] < 0:
                break
            candidate_food = int(self.pos_to_food[pos])
            candidate_cat = get_food_category(candidate_food)

            remaining_foods = [f for f in current_individual if f != food_id]
            remaining_cats = set(get_food_category(f) for f in remaining_foods)
            remaining_cats.add(candidate_cat)

            if len(remaining_cats) < len(current_cats) and len(current_cats) < max(4, self.max_categories // 12):
                continue

            new_total = current_without + self.matrix[candidate_food][col_idx]
            convergence = -np.sum((new_total - target_center) ** 2)

            div_bonus = 0.3 if candidate_cat not in current_cats else 0
            meal_bonus = 0.4 if self.meal_tags[candidate_food] in missing_meals else 0
            score = convergence * (1.0 + div_bonus + meal_bonus)

            if score > best_score:
                best_score = score
                best_food = candidate_food

        return int(best_food)

    def _expand_individual(self, individual, target_array, col_idx, min_size=15):
        """在保持可行的前提下，增加食物个数"""
        individual = np.array(individual, dtype=np.int32)

        while len(individual) < min_size:
            current_cats = set(get_food_category(fid) for fid in individual)
            missing_cats = set(self.category_list) - current_cats

            # 优先加新品类
            if missing_cats:
                chosen_cat = np.random.choice(list(missing_cats))
                cat_positions = self.category_to_positions[chosen_cat]
                new_food = int(self.pos_to_food[np.random.choice(cat_positions)])
            else:
                new_food = int(np.random.choice(self.all_valid_idx))

            if len(individual) < MAX_FOODS:
                individual = np.unique(np.concatenate([individual, [new_food]]))
            else:
                break

        return individual.astype(np.int32)

    def _local_search_2opt(self, individual, target_array, col_idx, max_iter=10):
        individual = np.array(individual, dtype=np.int32)
        if len(individual) < MIN_FOODS:
            return individual

        best_ind = individual.copy()
        best_fit, _, best_feas, _ = self._evaluate_single(best_ind, target_array, col_idx)

        for _ in range(max_iter):
            if len(best_ind) == 0:
                break
            pos = np.random.randint(0, len(best_ind))
            old_food = int(best_ind[pos])

            candidates = np.random.choice(self.all_valid_idx, size=min(10, len(self.all_valid_idx)), replace=False)
            for cand in candidates:
                cand = int(cand)
                if cand == old_food:
                    continue
                new_ind = best_ind.copy()
                new_ind[pos] = cand
                new_ind = np.unique(new_ind)
                fit, _, feas, _ = self._evaluate_single(new_ind, target_array, col_idx)
                if feas and fit > best_fit:
                    best_fit = fit
                    best_ind = new_ind
                    best_feas = True
                    break
            if best_feas and best_fit > 0.5:
                break

        return best_ind.astype(np.int32)

    def _evaluate_single(self, individual, target_array, col_idx):
        individual = np.array(individual, dtype=np.int32)
        n_foods = len(individual)
        if n_foods < MIN_FOODS or n_foods > MAX_FOODS:
            return 0.0, None, False, 0.0

        total = self.matrix[individual][:, col_idx].sum(axis=0)
        lo, hi = target_array[:, 0], target_array[:, 1]
        below = np.maximum(0, lo - total)
        above = np.maximum(0, total - hi)
        nutrient_penalty = float((below + above).sum())
        nutrient_feasible = nutrient_penalty < 1e-6

        meals = self.meal_tags[individual]
        meal_coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
        meal_penalty = float((3 - meal_coverage) * MEAL_COVERAGE_WEIGHT)

        categories = set()
        for fid in individual:
            categories.add(get_food_category(fid))
        diversity_score = len(categories) / self.max_categories if self.max_categories > 0 else 0.0
        diversity_penalty = float((1 - diversity_score) * DIVERSITY_WEIGHT)

        total_penalty = nutrient_penalty + meal_penalty + diversity_penalty
        fitness = 1.0 / (1.0 + max(0.0, total_penalty))
        feasible = nutrient_feasible and (meal_coverage == 3)

        return fitness, total, feasible, diversity_score

    def _initialize_population(self, pop_size=2000):
        population = []

        for _ in range(pop_size):
            n_cats = min(3, len(self.category_list))
            selected_cat_idx = np.random.choice(len(self.category_list), size=n_cats, replace=False)

            individual = []
            for ci in selected_cat_idx:
                cat_positions = self.category_pos_arrays[ci]
                cat_nutrients = self.matrix[self.pos_to_food[cat_positions]]
                if cat_nutrients.shape[1] > 0 and len(cat_positions) > 0:
                    variances = np.var(cat_nutrients, axis=1)
                    top_var_idx = np.argmax(variances)
                    individual.append(int(self.pos_to_food[cat_positions[top_var_idx]]))
                elif len(cat_positions) > 0:
                    individual.append(int(self.pos_to_food[np.random.choice(cat_positions)]))

            for meal in [0, 1, 2]:
                if not any(self.meal_tags[f] == meal for f in individual):
                    meal_pos = self.meal_positions[meal]
                    if len(meal_pos) > 0:
                        individual.append(int(self.pos_to_food[np.random.choice(meal_pos)]))

            # 强制至少15个食物
            while len(individual) < 15:
                individual.append(int(np.random.choice(self.all_valid_idx)))

            # 随机补充到 15-30
            n_extra = np.random.randint(0, MAX_FOODS - len(individual) + 1)
            if n_extra > 0:
                pool = []
                for ci in selected_cat_idx:
                    pool.extend(self.category_pos_arrays[ci])
                if len(pool) > 0:
                    pool_arr = np.array(pool, dtype=np.int32)
                    n_sample = min(n_extra, len(pool_arr))
                    if n_sample > 0:
                        extra_pos = np.random.choice(pool_arr, size=n_sample, replace=False)
                        individual.extend(int(self.pos_to_food[p]) for p in extra_pos)

            population.append(np.unique(np.array(individual, dtype=np.int32)))

        return population

    def _crossover(self, parent1, parent2):
        p1 = np.array(parent1, dtype=np.int32)
        p2 = np.array(parent2, dtype=np.int32)

        p1_cats = set(get_food_category(fid) for fid in p1)
        p2_cats = set(get_food_category(fid) for fid in p2)

        unique_p1 = [int(f) for f in p1 if get_food_category(f) not in p2_cats]
        unique_p2 = [int(f) for f in p2 if get_food_category(f) not in p1_cats]
        shared = [int(f) for f in np.unique(np.concatenate([p1, p2]))
                  if get_food_category(f) in p1_cats and get_food_category(f) in p2_cats]

        combined = np.unique(np.array(unique_p1 + unique_p2 + shared, dtype=np.int32))

        if len(combined) < MIN_FOODS:
            return p1.copy()

        if len(combined) > MAX_FOODS:
            selected = []
            seen_cats = set()
            for f in combined:
                cat = get_food_category(f)
                if cat not in seen_cats:
                    selected.append(int(f))
                    seen_cats.add(cat)

            # 强制至少15个
            remaining = max(15, MAX_FOODS) - len(selected)
            if remaining > 0:
                pool = [int(f) for f in combined if f not in selected]
                if pool:
                    n_add = min(remaining, len(pool))
                    selected.extend(np.random.choice(pool, size=n_add, replace=False).tolist())

            return np.array(selected, dtype=np.int32)
        else:
            # 如果不够15，补充
            if len(combined) < 15:
                pool = [int(f) for f in self.all_valid_idx if f not in combined]
                n_add = min(15 - len(combined), len(pool))
                if n_add > 0:
                    extra = np.random.choice(pool, size=n_add, replace=False)
                    combined = np.unique(np.concatenate([combined, extra]))
            return combined.copy()

    def _mutate_adaptive(self, individual, target_array, col_idx, gradient, gen, n_generations):
        individual = np.array(individual, dtype=np.int32)
        if len(individual) == 0:
            return individual

        # 强制增加食物个数（如果太少）
        if len(individual) < 15 and np.random.random() < 0.5:
            missing_cats = set(self.category_list) - set(get_food_category(fid) for fid in individual)
            if missing_cats:
                chosen_cat = np.random.choice(list(missing_cats))
                cat_positions = self.category_to_positions[chosen_cat]
                new_food = int(self.pos_to_food[np.random.choice(cat_positions)])
            else:
                new_food = int(np.random.choice(self.all_valid_idx))

            if len(individual) < MAX_FOODS:
                individual = np.unique(np.concatenate([individual, [new_food]]))
                return individual.astype(np.int32)

        ratio = gen / n_generations
        if ratio < 0.2:
            exploration = 0.8
        elif ratio < 0.6:
            exploration = 0.5
        else:
            exploration = 0.2

        mutated = individual.copy()
        current_cats = set(get_food_category(fid) for fid in mutated)
        min_cats_required = max(4, self.max_categories // 12)
        force_diversity = len(current_cats) < min_cats_required

        n_mutations = min(np.random.randint(1, 4), len(mutated))
        positions = np.random.choice(len(mutated), size=n_mutations, replace=False)

        for pos in positions:
            if np.random.random() < exploration:
                if force_diversity:
                    missing_cats = set(self.category_list) - current_cats
                    if missing_cats:
                        chosen_cat = np.random.choice(list(missing_cats))
                        cat_positions = self.category_to_positions[chosen_cat]
                        new_food = int(self.pos_to_food[np.random.choice(cat_positions)])
                        mutated[pos] = new_food
                        current_cats.add(chosen_cat)
                        force_diversity = len(current_cats) < min_cats_required
                    else:
                        mutated[pos] = int(np.random.choice(self.all_valid_idx))
                else:
                    mutated[pos] = int(np.random.choice(self.all_valid_idx))
            else:
                if force_diversity and np.random.random() < 0.5:
                    missing_cats = set(self.category_list) - current_cats
                    if missing_cats:
                        chosen_cat = np.random.choice(list(missing_cats))
                        cat_positions = self.category_to_positions[chosen_cat]
                        new_food = int(self.pos_to_food[np.random.choice(cat_positions)])
                        mutated[pos] = new_food
                        current_cats.add(chosen_cat)
                        force_diversity = len(current_cats) < min_cats_required
                    else:
                        new_food = self._vector_guided_replace(
                            int(mutated[pos]), mutated, target_array, col_idx, gradient, top_k=30
                        )
                        mutated[pos] = int(new_food)
                else:
                    new_food = self._vector_guided_replace(
                        int(mutated[pos]), mutated, target_array, col_idx, gradient, top_k=30
                    )
                    mutated[pos] = int(new_food)

        return np.unique(mutated).astype(np.int32)

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS):
        n_evals = 0

        self._precompute_structure()
        population = self._initialize_population(pop_size)

        scores, feasibles, diversities, n_foods_arr = evaluate_batch(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
        n_evals += pop_size

        if self.debug:
            if feasibles.sum() > 0:
                init_div = float(diversities[feasibles].mean())
                init_nf = float(n_foods_arr[feasibles].mean())
            else:
                init_div = 0.0
                init_nf = 0.0
            print(
                f"      [NutriSwap] Init: feasible={feasibles.sum()}/{pop_size}, best={scores.max():.4f}, div={init_div:.3f}, nf={init_nf:.1f}")

        feasible_archive = []
        best_feasible_fitness = 0.0
        best_feasible_individual = None
        ffg = None
        feasible_found = False
        last_feasible_gen = -1
        gradient = self._compute_dynamic_gradient(target_array, col_idx, population)

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break

            n_feas = feasibles.sum()

            for i in range(len(population)):
                if feasibles[i]:
                    feasible_archive.append((population[i].copy(), scores[i], diversities[i], n_foods_arr[i]))
                    if scores[i] > best_feasible_fitness:
                        best_feasible_fitness = scores[i]
                        best_feasible_individual = population[i].copy()
                    last_feasible_gen = gen

            if len(feasible_archive) > 100:
                feasible_archive.sort(key=lambda x: x[1], reverse=True)
                feasible_archive = feasible_archive[:100]

            if n_feas > 0 and not feasible_found:
                ffg = gen + 1
                feasible_found = True
                if self.debug:
                    feas_div = float(diversities[feasibles].mean()) if n_feas > 0 else 0.0
                    feas_nf = float(n_foods_arr[feasibles].mean()) if n_feas > 0 else 0.0
                    print(
                        f"      [NutriSwap] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}, div={feas_div:.3f}, nf={feas_nf:.1f}")

            if n_feas >= 20:
                if self.debug:
                    feas_div = float(diversities[feasibles].mean()) if n_feas > 0 else 0.0
                    feas_nf = float(n_foods_arr[feasibles].mean()) if n_feas > 0 else 0.0
                    print(
                        f"      [NutriSwap] Gen {gen + 1}: Early stop, {n_feas} feasible, div={feas_div:.3f}, nf={feas_nf:.1f}")
                break

            if gen - last_feasible_gen > 10 and not feasible_found:
                if self.debug:
                    print(f"      [NutriSwap] Gen {gen + 1}: RESTART 50%")
                n_restart = pop_size // 2
                for i in range(n_restart):
                    population[i] = self._initialize_population(1)[0]
                scores, feasibles, diversities, n_foods_arr = evaluate_batch(
                    population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
                n_evals += n_restart
                last_feasible_gen = gen
                gradient = self._compute_dynamic_gradient(target_array, col_idx, population)
                continue

            if gen % 3 == 0:
                gradient = self._compute_dynamic_gradient(target_array, col_idx, population)

            feas_idx = np.where(feasibles)[0]
            elite_size = min(len(feas_idx), pop_size // 3)
            new_pop = [population[i].copy() for i in feas_idx[:elite_size]]

            if gen % 10 == 0 and elite_size > 0:
                for i in range(min(2, elite_size)):
                    improved = self._local_search_2opt(new_pop[i], target_array, col_idx, max_iter=5)
                    new_pop[i] = improved

            n_diverse = int(pop_size * 0.3)
            diverse_pop = []
            for cat, positions in self.category_to_positions.items():
                if len(diverse_pop) >= n_diverse:
                    break
                if len(positions) > 0:
                    cat_foods = self.pos_to_food[np.array(positions)]
                    available = len(cat_foods)
                    if available >= MIN_FOODS:
                        n_items = np.random.randint(MIN_FOODS, min(MAX_FOODS, available) + 1)
                    elif available > 0:
                        n_items = available
                    else:
                        continue
                    ind = np.random.choice(cat_foods, size=min(n_items, available), replace=False)
                    for meal in [0, 1, 2]:
                        meal_pos = self.meal_positions[meal]
                        if len(meal_pos) > 0 and not np.any(self.meal_tags[ind] == meal):
                            ind = np.concatenate([ind, [self.pos_to_food[np.random.choice(meal_pos)]]])
                    diverse_pop.append(np.unique(ind.astype(np.int32)))

            new_pop.extend(diverse_pop[:n_diverse])

            if gen % 5 == 0 and len(feasible_archive) > 0 and len(new_pop) < pop_size:
                pop_cats = set()
                for ind in new_pop:
                    for fid in ind:
                        pop_cats.add(get_food_category(fid))

                diverse_candidates = []
                for archive_ind, archive_fit, archive_div, archive_nf in feasible_archive:
                    archive_cats = set(get_food_category(fid) for fid in archive_ind)
                    new_cats = len(archive_cats - pop_cats)
                    if new_cats > 0:
                        diverse_candidates.append((archive_ind, new_cats, archive_fit))

                diverse_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                n_inject = min(len(diverse_candidates), pop_size // 15)
                for i in range(n_inject):
                    new_pop.append(diverse_candidates[i][0].copy())

                if len(new_pop) < pop_size:
                    remaining = pop_size - len(new_pop)
                    other = [a for a in feasible_archive if a[0] not in [d[0] for d in diverse_candidates]]
                    if other:
                        n_more = min(remaining, len(other))
                        idx = np.random.choice(len(other), size=n_more, replace=False)
                        for i in idx:
                            new_pop.append(other[i][0].copy())

            remaining = pop_size - len(new_pop)
            if remaining > 0:
                infeas_idx = np.where(~feasibles)[0]
                if len(infeas_idx) > 0:
                    infeas_scores = np.maximum(scores[infeas_idx], 1e-10)
                    probs = infeas_scores / infeas_scores.sum()
                    chosen = np.random.choice(infeas_idx, size=min(remaining, len(infeas_idx)),
                                              p=probs, replace=True)
                    new_pop.extend([population[i].copy() for i in chosen])

                while len(new_pop) < pop_size:
                    new_pop.append(np.random.choice(self.all_valid_idx,
                                                    size=np.random.randint(MIN_FOODS, MAX_FOODS + 1), replace=False))

            population = new_pop[:pop_size]

            for i in range(elite_size, pop_size - 1, 2):
                if np.random.random() < 0.45:
                    population[i] = self._crossover(population[i], population[i + 1])

            for i in range(pop_size):
                if np.random.random() < 0.9:
                    population[i] = self._mutate_adaptive(
                        population[i], target_array, col_idx, gradient, gen, n_generations
                    )

            scores, feasibles, diversities, n_foods_arr = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
            n_evals += pop_size

        final_scores, final_feasibles, final_diversities, final_n_foods = evaluate_batch(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)

        if final_feasibles.sum() == 0 and best_feasible_individual is not None:
            if self.debug:
                print(f"      [NutriSwap] Restoring best feasible")
            population[0] = best_feasible_individual
            final_scores, final_feasibles, final_diversities, final_n_foods = evaluate_batch(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)

        if final_feasibles.sum() > 0:
            avg_diversity = float(final_diversities[final_feasibles].mean())
            avg_n_foods = float(final_n_foods[final_feasibles].mean())
        else:
            avg_diversity = 0.0
            avg_n_foods = 0.0

        if self.debug:
            print(
                f"      [NutriSwap] Final: feasible={final_feasibles.sum()}, best={final_scores.max():.4f}, div={avg_diversity:.3f}, nf={avg_n_foods:.1f}")

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
            'diversity': avg_diversity,
            'avg_n_foods': avg_n_foods,
        }]

