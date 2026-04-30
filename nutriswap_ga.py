"""
NutriSwap-GA: Nutrient-Semantic Swap Genetic Algorithm
基于营养指纹相似度的饮食组合遗传算法
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class NutriSwapGA:
    """
    饮食组合遗传算法

    核心创新:
    1. 指纹相似食物平替变异: 基于PCA降维的营养指纹 + KNN相似度索引
    2. 餐次结构保持交叉: 整餐交换确保推荐结果的现实合理性
    3. 锦标赛选择 + 精英保留

    个体编码: {早餐:[food_ids], 午餐:[food_ids], 晚餐:[food_ids]}
    交叉操作: 交换餐次（如交换两个组合的早餐）
    变异操作: 指纹相似的食物平替
    """

    def __init__(self, food_database, food_names, food_fingerprints=None):
        self.food_db = food_database
        self.food_names = food_names
        self.food_ids = list(food_database.keys())

        # 计算或加载指纹
        if food_fingerprints is None:
            self.food_fingerprints = self._compute_fingerprints()
        else:
            self.food_fingerprints = food_fingerprints

        self._build_similarity_index()

        self.meal_categories = {
            'breakfast': [],
            'lunch': [],
            'dinner': []
        }

    def _compute_fingerprints(self):
        """基于PCA计算每个食物的营养指纹"""
        food_matrix = np.array([self.food_db[fid] for fid in self.food_ids])
        scaler = StandardScaler()
        normalized = scaler.fit_transform(food_matrix)
        n_components = min(5, len(self.food_ids), food_matrix.shape[1])
        pca = PCA(n_components=n_components)
        fingerprints = pca.fit_transform(normalized)
        return {fid: fp for fid, fp in zip(self.food_ids, fingerprints)}

    def _build_similarity_index(self):
        """为每个食物建立KNN相似食物索引"""
        self.similar_foods = {}
        all_fps = np.array([self.food_fingerprints[fid] for fid in self.food_ids])
        nn = NearestNeighbors(n_neighbors=min(20, len(self.food_ids)), metric='euclidean')
        nn.fit(all_fps)

        for i, fid in enumerate(self.food_ids):
            n_neighbors = min(11, len(self.food_ids))
            distances, indices = nn.kneighbors([all_fps[i]], n_neighbors=n_neighbors)
            similar = [(self.food_ids[idx], 1/(1+d)) 
                      for d, idx in zip(distances[0][1:], indices[0][1:])]
            self.similar_foods[fid] = similar

    def categorize_foods(self, breakfast_ids, lunch_ids, dinner_ids):
        """将食物分类到不同餐次"""
        self.meal_categories['breakfast'] = breakfast_ids
        self.meal_categories['lunch'] = lunch_ids
        self.meal_categories['dinner'] = dinner_ids

    def create_random_daily_meal(self, breakfast_size=3, lunch_size=3, dinner_size=3):
        """随机生成一天的饮食组合"""
        meal = {}
        meal['breakfast'] = np.random.choice(
            self.meal_categories['breakfast'], size=breakfast_size, replace=True
        ).tolist()
        meal['lunch'] = np.random.choice(
            self.meal_categories['lunch'], size=lunch_size, replace=True
        ).tolist()
        meal['dinner'] = np.random.choice(
            self.meal_categories['dinner'], size=dinner_size, replace=True
        ).tolist()
        meal['total_nutrients'] = self._calculate_total_nutrients(meal)
        return meal

    def _calculate_total_nutrients(self, meal):
        """计算一天饮食的总营养素"""
        all_foods = meal['breakfast'] + meal['lunch'] + meal['dinner']
        total = np.zeros(len(self.food_db[all_foods[0]]))
        for food_id in all_foods:
            total += self.food_db[food_id]
        return total

    def evaluate_meal(self, meal, target_ranges):
        """
        评估饮食组合是否满足目标范围

        Returns:
            fitness: 1.0 / (1.0 + total_penalty), 范围(0, 1]
            total_nutrients: 总营养素向量
            nutrient_status: 每个营养素的状态
        """
        total_nutrients = meal['total_nutrients']
        total_penalty = 0
        nutrient_status = {}

        for idx, (lo, hi) in target_ranges.items():
            val = total_nutrients[idx]
            if val < lo:
                penalty = (lo - val) / (hi - lo)
                total_penalty += penalty
                nutrient_status[idx] = ('below', penalty)
            elif val > hi:
                penalty = (val - hi) / (hi - lo)
                total_penalty += penalty
                nutrient_status[idx] = ('above', penalty)
            else:
                nutrient_status[idx] = ('ok', 0)

        fitness = 1.0 / (1.0 + total_penalty)
        return fitness, total_nutrients, nutrient_status

    def crossover_meal_plan(self, parent1, parent2, method='by_meal'):
        """
        交叉两个一天的饮食组合

        method:
            'by_meal': 整餐交换（交换1-2餐）
            'by_meal_random': 每餐独立随机选择
        """
        child = {'breakfast': [], 'lunch': [], 'dinner': []}

        if method == 'by_meal':
            n_swap = np.random.randint(1, 3)
            meals_to_swap = np.random.choice(
                ['breakfast', 'lunch', 'dinner'], size=n_swap, replace=False
            )
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                if meal_type in meals_to_swap:
                    child[meal_type] = parent2[meal_type].copy()
                else:
                    child[meal_type] = parent1[meal_type].copy()

        elif method == 'by_meal_random':
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                if np.random.random() < 0.5:
                    child[meal_type] = parent1[meal_type].copy()
                else:
                    child[meal_type] = parent2[meal_type].copy()

        child['total_nutrients'] = self._calculate_total_nutrients(child)
        return child

    def mutate_by_similar_replacement(self, meal, target_ranges, 
                                     mutation_rate=0.3, max_replacements=2):
        """
        基于指纹相似度的变异
        在某餐中，用指纹相似的食物替换某个饮食项

        策略: 选择能使总营养更接近目标的食物
        """
        mutated_meal = {
            'breakfast': meal['breakfast'].copy(),
            'lunch': meal['lunch'].copy(),
            'dinner': meal['dinner'].copy()
        }

        replacements_made = 0
        mutation_log = []

        for meal_type in ['breakfast', 'lunch', 'dinner']:
            meal_items = mutated_meal[meal_type]

            for i, food_id in enumerate(meal_items):
                if replacements_made >= max_replacements:
                    break

                if np.random.random() < mutation_rate:
                    similar_foods = self.similar_foods[food_id]

                    if similar_foods:
                        new_food_id = self._select_best_replacement(
                            food_id, mutated_meal, i, meal_type, similar_foods, target_ranges
                        )

                        if new_food_id != food_id:
                            mutated_meal[meal_type][i] = new_food_id
                            replacements_made += 1
                            sim_val = next((s for fid, s in similar_foods if fid == new_food_id), 0)
                            mutation_log.append({
                                'meal_type': meal_type,
                                'position': i,
                                'old_food': self.food_names[food_id],
                                'new_food': self.food_names[new_food_id],
                                'similarity': sim_val
                            })

        mutated_meal['total_nutrients'] = self._calculate_total_nutrients(mutated_meal)
        mutated_meal['mutation_log'] = mutation_log
        return mutated_meal

    def _select_best_replacement(self, original_food_id, current_meal, 
                                 position, meal_type, similar_foods, target_ranges):
        """从相似食物中选择最佳替换"""
        current_total = self._calculate_total_nutrients(current_meal)
        current_total_without = current_total - self.food_db[original_food_id]

        best_food = original_food_id
        best_score = -np.inf

        target_center = np.array([(lo + hi) / 2 for lo, hi in target_ranges.values()])

        for similar_food_id, similarity in similar_foods[:10]:
            new_total = current_total_without + self.food_db[similar_food_id]
            distance = np.sum((new_total - target_center) ** 2)
            score = -distance + similarity * 0.1

            if score > best_score:
                best_score = score
                best_food = similar_food_id

        return best_food

    def run_evolution(self, target_ranges, n_generations=20, pop_size=20, 
                      crossover_method='by_meal', mutation_rate=0.4,
                      tournament_size=3, elitism=True, verbose=True):
        """
        运行完整进化过程

        Returns:
            best_meal: 最佳饮食组合
            history: 每代统计信息
        """
        # 初始化种群
        pop = []
        for _ in range(5):
            pop.append(self.create_random_daily_meal())

        history = []

        for gen in range(n_generations):
            # 评估
            scores = []
            nutrients_list = []
            for ind in pop:
                fitness, nutrients, _ = self.evaluate_meal(ind, target_ranges)
                scores.append(fitness)
                nutrients_list.append(nutrients)

            scores = np.array(scores)
            nutrients_arr = np.array(nutrients_list)

            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)

            gen_info = {
                'generation': gen + 1,
                'best_fitness': scores[best_idx],
                'worst_fitness': scores[worst_idx],
                'avg_fitness': np.mean(scores),
                'std_fitness': np.std(scores),
                'best_nutrients': nutrients_arr[best_idx],
                'n_perfect': sum(1 for s in scores if s >= 0.999)
            }
            history.append(gen_info)

            if verbose and (gen % 5 == 0 or gen == n_generations - 1):
                print(f"  Gen {gen+1:2d}: best={scores[best_idx]:.4f} avg={np.mean(scores):.4f} "
                      f"worst={scores[worst_idx]:.4f} perfect={gen_info['n_perfect']}/{len(pop)}")

            # 生成新种群
            best = pop[best_idx]
            new_pop = [best] if elitism else []

            while len(new_pop) < pop_size:
                # 锦标赛选择
                candidates = np.random.choice(len(pop), tournament_size, replace=False)
                w1 = candidates[np.argmax([scores[c] for c in candidates])]
                candidates = np.random.choice(len(pop), tournament_size, replace=False)
                w2 = candidates[np.argmax([scores[c] for c in candidates])]

                # 交叉
                child = self.crossover_meal_plan(pop[w1], pop[w2], method=crossover_method)

                # 变异
                if np.random.random() < mutation_rate:
                    child = self.mutate_by_similar_replacement(
                        child, target_ranges, mutation_rate=0.3, max_replacements=2
                    )

                new_pop.append(child)

            pop = new_pop

        # 返回最佳个体
        final_scores = []
        for ind in pop:
            fitness, _, _ = self.evaluate_meal(ind, target_ranges)
            final_scores.append(fitness)

        best_idx = np.argmax(final_scores)
        return pop[best_idx], history
