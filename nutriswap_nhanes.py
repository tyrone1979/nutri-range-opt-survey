"""
NutriSwap-GA v2: 统一编码 + 餐覆盖约束 + 智能初始化 + 保留所有可行解
目标: NutriSwap-GA 超越 BMOEA

核心变更:
  1. 统一食物池编码（取消三餐分离），约束中强制餐类型覆盖=3
  2. NutriSwap保留所有可行解（类似BMOEA精英保留）
  3. 修复交叉和变异算子（BMOEA式合并重采样）
  4. 智能初始化: 随机生成 → 全部不可行时启动修复逻辑
  5. NSGA2/NSGA3/MOEAD/RVEA 用 pymoo（避免自研bug）
"""

import json
import os
import time
import warnings
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ====================== 全局参数 ======================
MIN_FOODS = 3      # 最少食物数（每餐至少1个 × 3餐）
MAX_FOODS = 30     # 最多食物数（每餐最多10个 × 3餐）
POP_SIZE = 5000
N_GENERATIONS = 50
MAX_EVALS = 500000
MEAL_COVERAGE_WEIGHT = 100.0  # 餐覆盖缺失惩罚权重

EATING_TYPE_MAP = {
    1:'breakfast',2:'lunch',3:'dinner',4:'dinner',5:'breakfast',
    6:'lunch',7:'lunch',8:'breakfast',9:'lunch',10:'breakfast',
    11:'lunch',12:'dinner',13:'lunch',14:'dinner',15:'lunch',
    16:'lunch',17:'lunch',18:'lunch',19:'lunch',91:'lunch',
}

# ============================================================
# 1. 数据加载
# ============================================================

def load_data(nhanes_path, food_type_path, target_path, missing_threshold=0.5):
    """加载nhanes数据，返回matrix和每个食物的餐类型标签"""
    print("[1/4] Loading nhanes.csv...")
    nhanes = pd.read_csv(nhanes_path, low_memory=False, na_values=['', ' ', 'NA'])
    print(f"       Rows: {len(nhanes)}")

    print("[2/4] Loading food_type.csv...")
    ft = pd.read_csv(food_type_path)
    ft_best = ft.groupby('food_id')['eating_type'].agg(lambda x: x.value_counts().index[0]).reset_index()
    ft_best['meal_type'] = ft_best['eating_type'].map(EATING_TYPE_MAP)
    ft_map = dict(zip(ft_best['food_id'].astype(str), ft_best['meal_type']))
    nhanes['meal_type'] = nhanes['food_id'].astype(str).map(ft_map)
    nhanes = nhanes[nhanes['meal_type'].notna()].copy()
    print(f"       After meal map: {len(nhanes)}")

    # 营养素列
    exclude = {'food_id', 'grams', 'meal_type'}
    nutrients = [c for c in nhanes.columns if c not in exclude]
    for c in nutrients:
        nhanes[c] = pd.to_numeric(nhanes[c], errors='coerce')
    mr = nhanes[nutrients].isnull().mean()
    drop_cols = mr[mr > missing_threshold].index.tolist()
    if drop_cols:
        nhanes = nhanes.drop(columns=drop_cols)
        nutrients = [c for c in nutrients if c not in drop_cols]
    nhanes[nutrients] = nhanes[nutrients].fillna(0)
    print(f"       Nutrients ({len(nutrients)}): {nutrients}")

    # 餐类型编码: 0=breakfast, 1=lunch, 2=dinner
    meal_type_map = {'breakfast': 0, 'lunch': 1, 'dinner': 2}
    meal_tags = nhanes['meal_type'].map(meal_type_map).values.astype(np.int8)

    print("[3/4] Loading target.csv...")
    target_df = pd.read_csv(target_path)
    all_targets = []
    for idx in range(len(target_df)):
        try:
            td = eval(target_df.iloc[idx, 0])
        except Exception:
            continue
        common = [n for n in nutrients if n in td]
        if len(common) == 0:
            continue
        col_idx = np.array([nutrients.index(n) for n in common], dtype=np.int32)
        target_arr = np.array([[td[n][0], td[n][1]] for n in common], dtype=np.float32)
        all_targets.append({
            'idx': idx, 'nutrients': common,
            'col_idx': col_idx, 'target_array': target_arr,
        })
    print(f"       Valid targets: {len(all_targets)}")

    matrix = nhanes[nutrients].values.astype(np.float32)
    print(f"[4/4] Matrix: {matrix.shape}, meal_tags: {meal_tags.shape}")
    print(f"       b/l/d counts: {(meal_tags==0).sum()}/{(meal_tags==1).sum()}/{(meal_tags==2).sum()}")
    return matrix, meal_tags, nutrients, all_targets


# ============================================================
# 2. 工具函数
# ============================================================

def prefilter(matrix, target_array, col_idx):
    """预过滤: 排除超过任何营养素上限的食物"""
    max_vals = target_array[:, 1]
    valid = np.all(matrix[:, col_idx] <= max_vals, axis=1)
    return np.where(valid)[0]


def evaluate(individual, matrix, target_array, col_idx, meal_tags):
    """
    评估单个个体（食物索引数组）
    返回: (fitness, total_nutrients, is_feasible)
    约束: 营养素在范围内 AND 餐类型覆盖=3
    """
    n_foods = len(individual)
    if n_foods < MIN_FOODS or n_foods > MAX_FOODS:
        return 0.0, None, False

    # 营养素总和
    total = matrix[individual][:, col_idx].sum(axis=0)
    lo, hi = target_array[:, 0], target_array[:, 1]
    below = np.maximum(0, lo - total)
    above = np.maximum(0, total - hi)
    nutrient_penalty = (below + above).sum()
    nutrient_feasible = nutrient_penalty < 1e-6

    # 餐类型覆盖检查
    meals = meal_tags[individual]
    has_b = np.any(meals == 0)
    has_l = np.any(meals == 1)
    has_d = np.any(meals == 2)
    meal_coverage = int(has_b) + int(has_l) + int(has_d)
    meal_penalty = (3 - meal_coverage) * MEAL_COVERAGE_WEIGHT

    total_penalty = nutrient_penalty + meal_penalty
    fitness = 1.0 / (1.0 + total_penalty)
    feasible = nutrient_feasible and (meal_coverage == 3)

    return fitness, total, feasible


def fast_evaluate_pop(population, matrix, target_array, col_idx, meal_tags):
    """批量评估种群，返回 (scores_array, feasible_array)"""
    pop_size = len(population)
    scores = np.zeros(pop_size, dtype=np.float32)
    feasibles = np.zeros(pop_size, dtype=bool)
    for i in range(pop_size):
        f, _, fe = evaluate(population[i], matrix, target_array, col_idx, meal_tags)
        scores[i] = f
        feasibles[i] = fe
    return scores, feasibles


def random_individual(all_valid_idx, meal_tags):
    """生成随机个体（确保餐覆盖）"""
    # 先确保每种餐至少1个
    b_idx = all_valid_idx[meal_tags[all_valid_idx] == 0]
    l_idx = all_valid_idx[meal_tags[all_valid_idx] == 1]
    d_idx = all_valid_idx[meal_tags[all_valid_idx] == 2]

    if len(b_idx) == 0 or len(l_idx) == 0 or len(d_idx) == 0:
        return None

    individual = [
        np.random.choice(b_idx, 1, replace=True)[0],
        np.random.choice(l_idx, 1, replace=True)[0],
        np.random.choice(d_idx, 1, replace=True)[0],
    ]
    # 额外随机食物
    n_extra = np.random.randint(0, MAX_FOODS - 2)
    if n_extra > 0:
        extra = np.random.choice(all_valid_idx, n_extra, replace=True)
        individual.extend(extra)

    return np.array(individual, dtype=np.int32)



# ============================================================
# 4. NutriSwap-GA (核心算法)
# ============================================================
"""
NutriSwap-GA v12: 统一编码 + 指纹相似度智能变异（优化版）
优化:
  1. 使用随机采样构建指纹索引（不全量）
  2. 懒惰初始化：只在需要时构建
  3. 缓存相似食物映射
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


class NutriSwapGA:
    """
    NutriSwap-GA v12: 统一编码 + 指纹相似度智能变异（优化版）
    """

    def __init__(self, matrix, meal_tags, all_valid_idx, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.debug = debug

        # 懒惰初始化：只在第一次使用时构建
        self.similar_foods = None
        self._fingerprint_built = False

        if self.debug:
            print(f"      [NutriSwap] Initialized: {len(all_valid_idx)} foods")

    def _build_fingerprint_index_fast(self):
        """快速构建指纹索引（使用采样）"""
        if self._fingerprint_built:
            return

        n_foods = len(self.all_valid_idx)

        # 如果食物数量太多，随机采样
        if n_foods > 10000:
            sample_size = min(5000, n_foods)
            sample_indices = np.random.choice(n_foods, size=sample_size, replace=False)
            sampled_foods = self.all_valid_idx[sample_indices]
            sampled_matrix = self.matrix[sampled_foods]

            if self.debug:
                print(f"      [NutriSwap] Building index on {sample_size}/{n_foods} samples")
        else:
            sampled_foods = self.all_valid_idx
            sampled_matrix = self.matrix[sampled_foods]

        # 标准化
        scaler = StandardScaler()
        normalized = scaler.fit_transform(sampled_matrix)

        # PCA降维
        n_components = min(5, len(sampled_foods) - 1, sampled_matrix.shape[1])
        if n_components >= 2:
            pca = PCA(n_components=n_components)
            fingerprints = pca.fit_transform(normalized)
        else:
            fingerprints = normalized

        # 建立KNN索引（只对采样）
        n_neighbors = min(20, len(sampled_foods))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(fingerprints)

        # 为每个食物找到相似食物（只对采样）
        self.similar_foods = {}
        for i, fid in enumerate(sampled_foods):
            distances, indices = nn.kneighbors([fingerprints[i]], n_neighbors=n_neighbors)
            similar = []
            for d, idx in zip(distances[0][1:], indices[0][1:]):
                if d > 0:
                    similarity = 1.0 / (1.0 + d)
                    similar.append((sampled_foods[idx], similarity))
            self.similar_foods[fid] = similar

        # 对于未采样的食物，使用近似（找最近的采样点）
        if n_foods > 10000:
            # 计算未采样食物的指纹（用同一PCA模型）
            unsampled_foods = np.setdiff1d(self.all_valid_idx, sampled_foods)
            if len(unsampled_foods) > 0:
                unsampled_matrix = self.matrix[unsampled_foods]
                unsampled_normalized = scaler.transform(unsampled_matrix)
                if n_components >= 2:
                    unsampled_fingerprints = pca.transform(unsampled_normalized)
                else:
                    unsampled_fingerprints = unsampled_normalized

                # 找到每个未采样食物的最近采样点
                neighbors = nn.kneighbors(unsampled_fingerprints, n_neighbors=1)
                for i, fid in enumerate(unsampled_foods):
                    nearest_sample_idx = neighbors[1][i][0]
                    nearest_sample_fid = sampled_foods[nearest_sample_idx]
                    # 使用采样点的相似食物作为近似
                    self.similar_foods[fid] = self.similar_foods.get(nearest_sample_fid, [])

        self._fingerprint_built = True

        if self.debug:
            print(f"      [NutriSwap] Index built, coverage: {len(self.similar_foods)}/{n_foods}")

    def _get_similar_foods(self, food_id):
        """获取相似食物（懒惰构建）"""
        if not self._fingerprint_built:
            self._build_fingerprint_index_fast()
        return self.similar_foods.get(food_id, [])

    def _random_individual(self):
        """完全随机生成个体"""
        n_items = np.random.randint(3, min(30, len(self.all_valid_idx)) + 1)
        return np.random.choice(self.all_valid_idx, size=n_items, replace=False)

    def _select_best_replacement(self, original_food, current_individual, pos,
                                 target_array, col_idx):
        """从相似食物中选择最佳替换"""
        similar_foods = self._get_similar_foods(original_food)[:10]
        if not similar_foods:
            return original_food

        # 计算当前总营养（不含待替换食物）
        current_total = self.matrix[current_individual][:, col_idx].sum(axis=0)
        current_total_without = current_total - self.matrix[original_food][col_idx]

        # 目标中心点
        target_center = (target_array[:, 0] + target_array[:, 1]) / 2.0

        best_food = original_food
        best_score = -np.inf

        for similar_food, similarity in similar_foods:
            new_total = current_total_without + self.matrix[similar_food][col_idx]
            distance = np.sum((new_total - target_center) ** 2)
            score = -distance + similarity * 10.0

            if score > best_score:
                best_score = score
                best_food = similar_food

        return best_food

    def _mutate_smart(self, individual, target_array, col_idx):
        """智能变异"""
        if len(individual) == 0:
            return individual

        mutated = individual.copy()

        n_replace = min(np.random.randint(1, 3), len(mutated))
        positions = np.random.choice(len(mutated), size=n_replace, replace=False)

        for pos in positions:
            new_food = self._select_best_replacement(
                mutated[pos], mutated, pos, target_array, col_idx
            )
            mutated[pos] = new_food

        return np.unique(mutated)

    def _mutate_random(self, individual):
        """随机变异"""
        if len(individual) == 0:
            return individual

        mutated = individual.copy()
        action = np.random.randint(0, 3)

        if action == 0 and len(mutated) < 30:
            new_food = np.random.choice(self.all_valid_idx)
            mutated = np.unique(np.concatenate([mutated, [new_food]]))
        elif action == 1 and len(mutated) > 3:
            pos = np.random.randint(0, len(mutated))
            mutated = np.delete(mutated, pos)
        else:
            pos = np.random.randint(0, len(mutated))
            new_food = np.random.choice(self.all_valid_idx)
            mutated[pos] = new_food
            mutated = np.unique(mutated)

        return mutated

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS):

        n_evals = 0

        # 懒惰构建指纹索引（耗时操作延迟到第一次需要时）
        # 注意：这里不调用，只在变异时按需构建

        # 初始化种群
        population = []
        for _ in range(pop_size):
            population.append(self._random_individual())

        # 评估
        scores, feasibles = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags)
        n_evals += pop_size

        if self.debug:
            print(f"      [NutriSwap] Init: feasible={feasibles.sum()}/{pop_size}, best={scores.max():.4f}")

        # 进化循环
        ffg = None
        feasible_found = False
        elite_size = 0

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break

            n_feas = feasibles.sum()

            if n_feas > 0 and not feasible_found:
                ffg = gen + 1
                feasible_found = True
                if self.debug:
                    print(f"      [NutriSwap] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}")

            if n_feas >= 20:
                if self.debug:
                    print(f"      [NutriSwap] Gen {gen + 1}: Early stop, {n_feas} feasible")
                break

            # 精英保留
            feas_idx = np.where(feasibles)[0]
            num_valid = min(len(feas_idx), pop_size)
            elite_size = num_valid

            new_pop = [population[i].copy() for i in feas_idx[:num_valid]]

            remaining = pop_size - len(new_pop)
            if remaining > 0:
                infeas_idx = np.where(~feasibles)[0]
                if len(infeas_idx) > 0:
                    infeas_scores = np.maximum(scores[infeas_idx], 1e-10)
                    probs = infeas_scores / infeas_scores.sum()
                    chosen = np.random.choice(infeas_idx, size=remaining, p=probs, replace=True)
                    new_pop.extend([population[i].copy() for i in chosen])
                else:
                    for i in range(remaining):
                        new_pop.append(population[feas_idx[i % len(feas_idx)]].copy())

            population = new_pop[:pop_size]

            # 交叉
            for i in range(elite_size, pop_size - 1, 2):
                if np.random.random() < 0.2:
                    combined = np.unique(np.concatenate([population[i], population[i + 1]]))
                    if len(combined) >= 3:
                        n_target = np.random.randint(3, min(30, len(combined)) + 1)
                        population[i] = np.random.choice(combined, size=n_target, replace=False)

            # 变异（首次智能变异时会触发索引构建）
            for i in range(pop_size):
                if np.random.random() < 0.9:
                    if np.random.random() < 0.7:
                        # 智能变异（可能触发索引构建）
                        population[i] = self._mutate_smart(population[i], target_array, col_idx)
                    else:
                        population[i] = self._mutate_random(population[i])

            # 重新评估
            scores, feasibles = fast_evaluate_pop(
                population, self.matrix, target_array, col_idx, self.meal_tags)
            n_evals += pop_size

        final_scores, final_feasibles = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags)

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
        }]

# ============================================================
# 5. BMOEA (Baseline)
# ============================================================

class BMOEA:
    """BMOEA: 统一编码版本"""

    def __init__(self, matrix, meal_tags, all_valid_idx, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
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
        gen_feas_log = []
        elite_size = 0
        n_evals = 0

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break

            scores, feasibles = fast_evaluate_pop(
                population, self.matrix, target_array, col_idx, self.meal_tags)
            n_evals += pop_size

            n_feas = feasibles.sum()
            gen_feas_log.append(int(n_feas))

            if n_feas > 0 and not feasible_found:
                ffg = gen + 1
                feasible_found = True
                if self.debug:
                    print(f"      [BMOEA] Gen {gen+1}: FIRST FEASIBLE! count={n_feas}")

            if n_feas >= 20:
                if self.debug:
                    print(f"      [BMOEA] Gen {gen+1}: Early stop, {n_feas} feasible")
                break

            # 保留可行解
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

            # 交叉
            for i in range(elite_size, pop_size - 1, 2):
                if np.random.random() < crossover_rate:
                    combined = np.unique(np.concatenate([population[i], population[i + 1]]))
                    n_target = np.random.randint(MIN_FOODS, min(MAX_FOODS, len(combined)) + 1)
                    population[i] = np.random.choice(combined, size=n_target, replace=False)

            # 变异
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

        if self.debug:
            print(f"      [BMOEA] Gen feas (first 20): {gen_feas_log[:20]}")

        final_scores, final_feasibles = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags)

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
        }]


# ============================================================
# 6. ACMO-Diet
# ============================================================
class ACMODiet:
    """ACMO-Diet: 统一编码版本 - 修复版 v2"""

    def __init__(self, matrix, meal_tags, all_valid_idx, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.debug = debug

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS):

        # 确保有足够种群用于锦标赛选择
        population = []
        for _ in range(pop_size):
            # 改进：初始种群包含随机和餐覆盖两种
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

            scores, feasibles = fast_evaluate_pop(
                population, self.matrix, target_array, col_idx, self.meal_tags)
            n_evals += pop_size

            n_feas = feasibles.sum()

            # 记录首次可行
            if n_feas > 0 and ffg is None:
                ffg = gen + 1
                if self.debug:
                    print(f"      [ACMO] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}")

            # 【修复】每代都检查早停条件
            if n_feas >= 20:
                if self.debug:
                    print(f"      [ACMO] Gen {gen + 1}: Early stop, {n_feas} feasible")
                break

            # 保留最佳个体
            best_idx = np.argmax(scores)
            new_pop = [population[best_idx].copy()]

            while len(new_pop) < pop_size:
                # 锦标赛选择父本1
                tour1 = np.random.choice(len(population), 3, replace=False)
                parent1_idx = tour1[np.argmax(scores[tour1])]

                # 锦标赛选择父本2
                tour2 = np.random.choice(len(population), 3, replace=False)
                parent2_idx = tour2[np.argmax(scores[tour2])]

                child = population[parent1_idx].copy()

                # 交叉操作
                if np.random.random() < 0.5 and len(child) > 0:
                    parent2 = population[parent2_idx]
                    if len(parent2) > 0:
                        # 决定替换数量
                        n_replace = max(1, min(len(child) // 3, len(parent2)))
                        if n_replace <= len(child):
                            positions = np.random.choice(len(child), size=n_replace, replace=False)
                            # 从parent2中选择食物替换
                            replace_foods = np.random.choice(parent2, size=min(n_replace, len(parent2)), replace=True)
                            for pos, food in zip(positions, replace_foods[:len(positions)]):
                                child[pos] = food

                    # 去重
                    child = np.unique(child)

                # 变异操作
                if np.random.random() < 0.3 and len(child) > 0:
                    action = np.random.randint(0, 3)

                    if action == 0 and len(child) < MAX_FOODS:
                        # 添加食物
                        new_food = np.random.choice(self.all_valid_idx)
                        child = np.unique(np.concatenate([child, [new_food]]))

                    elif action == 1 and len(child) > MIN_FOODS:
                        # 删除食物
                        remove_pos = np.random.randint(0, len(child))
                        child = np.delete(child, remove_pos)

                    else:
                        # 替换食物
                        replace_pos = np.random.randint(0, len(child))
                        new_food = np.random.choice(self.all_valid_idx)
                        child[replace_pos] = new_food
                        child = np.unique(child)

                # 确保个体大小在合理范围
                if len(child) < MIN_FOODS:
                    # 补充食物
                    additional = np.random.choice(self.all_valid_idx, MIN_FOODS - len(child), replace=False)
                    child = np.unique(np.concatenate([child, additional]))

                if len(child) > MAX_FOODS:
                    # 随机删除多余食物
                    remove_count = len(child) - MAX_FOODS
                    remove_indices = np.random.choice(len(child), size=remove_count, replace=False)
                    child = np.delete(child, remove_indices)

                new_pop.append(child)

            population = new_pop

        # 最终评估
        final_scores, final_feasibles = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags)

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
        }]

# ============================================================
# 7. Random Search
# ============================================================

class RandomSearch:
    """Random Search: 统一编码版本"""

    def __init__(self, matrix, meal_tags, all_valid_idx):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS):

        ffg = None
        best_fitness = 0.0
        total_feas = 0
        n_evals = 0

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break
            for _ in range(pop_size):
                ind = random_individual(self.all_valid_idx, self.meal_tags)
                if ind is None:
                    ind = np.random.choice(self.all_valid_idx, MIN_FOODS, replace=False)
                f, _, fe = evaluate(ind, self.matrix, target_array, col_idx, self.meal_tags)
                if f > best_fitness:
                    best_fitness = f
                if fe:
                    total_feas += 1
                    if ffg is None:
                        ffg = gen + 1
                n_evals += 1

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if ffg else 0.0,
            'fsc': total_feas,
            'best_fitness': best_fitness,
            'n_evals': n_evals,
        }]


# ============================================================
# 8. pymoo 多目标算法适配器
# ============================================================

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2 as PyMOONSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3 as PyMOONSGA3
    from pymoo.algorithms.moo.moead import MOEAD as PyMOOMOEAD
    from pymoo.algorithms.moo.rvea import RVEA as PyMOORVEA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not available, multi-objective algorithms will be skipped")


class DietProblem(Problem):
    """pymoo Problem适配器: 离散食物选择优化"""

    def __init__(self, matrix, target_array, col_idx, meal_tags, all_valid_idx, n_obj=2):
        self.matrix = matrix
        self.target_array = target_array
        self.col_idx = col_idx
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.target_center = (target_array[:, 0] + target_array[:, 1]) / 2.0

        # 决策变量: 30个整数（食物索引或-1表示空）
        n_var = MAX_FOODS
        xl = np.zeros(n_var)
        xu = np.ones(n_var) * (len(all_valid_idx) - 1)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        n_ind = x.shape[0]
        F = np.zeros((n_ind, self.n_obj))

        for i in range(n_ind):
            # 解码: 只使用非-1的位置，映射回真实食物索引
            valid_mask = x[i] >= 0
            indices = self.all_valid_idx[x[i][valid_mask].astype(int)]

            if len(indices) < MIN_FOODS:
                F[i, :] = 1e6
                continue

            total = self.matrix[indices][:, self.col_idx].sum(axis=0)

            if self.n_obj == 2:
                # obj1: 营养素偏离
                below = np.maximum(0, self.target_array[:, 0] - total)
                above = np.maximum(0, total - self.target_array[:, 1])
                F[i, 0] = (below + above).sum()
                # obj2: 食物多样性（负数，最大化唯一食物数）
                F[i, 1] = -len(set(indices))
            elif self.n_obj == 3:
                # obj1-3: 各营养素偏离
                for j in range(min(3, len(self.col_idx))):
                    tc = self.target_center[j]
                    F[i, j] = abs(total[j] - tc)

        out["F"] = F


def run_pymoo_algorithm(algo_name, matrix, target_array, col_idx, meal_tags, all_valid_idx,
                        pop_size=500, n_generations=50):
    """运行pymoo算法"""
    if not PYMOO_AVAILABLE:
        return {'ffg': n_generations + 1, 'fhr': 0.0, 'fsc': 0, 'best_fitness': 0.0, 'n_evals': 0}

    n_obj = 2 if algo_name in ['NSGA2', 'NSGA3'] else 3
    problem = DietProblem(matrix, target_array, col_idx, meal_tags, all_valid_idx, n_obj=n_obj)

    if algo_name == 'NSGA2':
        algorithm = PyMOONSGA2(pop_size=pop_size)
    elif algo_name == 'NSGA3':
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        algorithm = PyMOONSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
    elif algo_name == 'MOEAD':
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        algorithm = PyMOOMOEAD(ref_dirs=ref_dirs, n_neighbors=15)
    elif algo_name == 'RVEA':
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        algorithm = PyMOORVEA(ref_dirs=ref_dirs, pop_size=pop_size)
    else:
        return {'ffg': n_generations + 1, 'fhr': 0.0, 'fsc': 0, 'best_fitness': 0.0, 'n_evals': 0}

    # 自定义初始化（确保餐覆盖）
    from pymoo.core.initialization import Initialization
    from pymoo.core.sampling import Sampling

    class DietSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            X = np.full((n_samples, problem.n_var), -1, dtype=int)
            for i in range(n_samples):
                ind = random_individual(all_valid_idx, meal_tags)
                if ind is not None:
                    # 将真实索引映射回all_valid_idx中的位置
                    for j, real_idx in enumerate(ind):
                        if j < problem.n_var:
                            pos = np.where(all_valid_idx == real_idx)[0]
                            if len(pos) > 0:
                                X[i, j] = pos[0]
            return X

    algorithm.initialization = Initialization(DietSampling())

    try:
        res = minimize(problem, algorithm, ('n_gen', n_generations), seed=42, verbose=False)

        # 从结果中提取最佳解
        best_fitness = 0.0
        n_feas = 0
        for sol in res.X:
            valid_mask = sol >= 0
            indices = all_valid_idx[sol[valid_mask].astype(int)]
            if len(indices) >= MIN_FOODS:
                f, _, fe = evaluate(indices, matrix, target_array, col_idx, meal_tags)
                if f > best_fitness:
                    best_fitness = f
                if fe:
                    n_feas += 1

        # pymoo不直接提供FFG，我们用n_evals估算
        n_evals = pop_size * n_generations

        return {
            'ffg': n_generations + 1,  # pymoo不提供逐代信息
            'fhr': 1.0 if n_feas > 0 else 0.0,
            'fsc': n_feas,
            'best_fitness': best_fitness,
            'n_evals': n_evals,
        }
    except Exception as e:
        print(f"      [pymoo {algo_name}] Error: {e}")
        return {'ffg': n_generations + 1, 'fhr': 0.0, 'fsc': 0, 'best_fitness': 0.0, 'n_evals': 0}


# ============================================================
# 9. 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("NutriSwap-GA v2: 统一编码 + 智能初始化 + 保留所有可行解")
    print("=" * 70)

    paths = {
        'nhanes': ['./data/nhanes.csv', '/mnt/agents/upload/nhanes.csv'],
        'food_type': ['./data/food_type.csv', '/mnt/agents/upload/food_type.csv'],
        'target': ['./data/target.csv', '/mnt/agents/upload/target.csv'],
    }
    found = {}
    for k, cs in paths.items():
        for p in cs:
            if os.path.exists(p):
                found[k] = p
                break
    if not all(k in found for k in ['nhanes', 'food_type', 'target']):
        print("Error: Missing data files!")
        exit(1)

    matrix, meal_tags, nutrients, all_targets = load_data(
        found['nhanes'], found['food_type'], found['target'])
    n_foods = len(matrix)
    n_targets = len(all_targets)
    print(f"\nTotal targets: {n_targets}")

    # 运行一个目标（第一个）
    target_info = all_targets[0]
    ta, ci = target_info['target_array'], target_info['col_idx']

    print(f"\n{'='*70}")
    print(f"Target 1/{n_targets} (idx={target_info['idx']})")
    print(f"Nutrients: {target_info['nutrients']}")
    print(f"Ranges: {ta.tolist()}")

    # 预过滤
    all_valid = prefilter(matrix, ta, ci)
    print(f"Pre-filtered: {len(all_valid)}/{n_foods} foods")

    # 检查每种餐类型的数量
    b_count = np.sum(meal_tags[all_valid] == 0)
    l_count = np.sum(meal_tags[all_valid] == 1)
    d_count = np.sum(meal_tags[all_valid] == 2)
    print(f"  b/l/d: {b_count}/{l_count}/{d_count}")

    if len(all_valid) < MIN_FOODS or b_count == 0 or l_count == 0 or d_count == 0:
        print("  Not enough valid foods or missing meal types, skipping")
        exit(1)

    # 随机可行率
    def _rand_ind():
        ind = random_individual(all_valid, meal_tags)
        if ind is None:
            ind = np.random.choice(all_valid, MIN_FOODS, replace=False)
        return ind
    fc = sum(1 for _ in range(100) if evaluate(_rand_ind(), matrix, ta, ci, meal_tags)[2])
    print(f"Random feasibility: {fc}%")

    # 运行算法
    print(f"\n{'='*70}")
    print(f"Running algorithms (pop={POP_SIZE}, gen={N_GENERATIONS})")
    print(f"{'Algorithm':<15} {'FFG':>6} {'FHR':>6} {'FSC':>6} {'BestFit':>8} {'Evals':>8} {'Time(s)':>8}")
    print(f"{'-'*65}")

    algorithms = {
        'NutriSwap': lambda: NutriSwapGA(matrix, meal_tags, all_valid, debug=True).run(ta, ci),
        'BMOEA': lambda: BMOEA(matrix, meal_tags, all_valid, debug=True).run(ta, ci),
        'ACMO-Diet': lambda: ACMODiet(matrix, meal_tags, all_valid, debug=True).run(ta, ci),
        'Random': lambda: RandomSearch(matrix, meal_tags, all_valid).run(ta, ci),
    }

    # pymoo算法（较小种群）
    if PYMOO_AVAILABLE:
        pymoo_algos = ['NSGA2', 'NSGA3', 'MOEAD', 'RVEA']
        for name in pymoo_algos:
            algorithms[name] = lambda n=name: [run_pymoo_algorithm(
                n, matrix, ta, ci, meal_tags, all_valid, pop_size=500, n_generations=50)]

    all_results = {}
    for name, algo_fn in algorithms.items():
        t0 = time.time()
        try:
            result = algo_fn()
            r = result[0]
            elapsed = time.time() - t0
            all_results[name] = r
            print(f"{name:<15} {r['ffg']:6.0f} {r['fhr']*100:5.0f}% {r['fsc']:5.0f} {r['best_fitness']:8.3f} {r['n_evals']:7.0f} {elapsed:7.1f}")
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")

    print(f"\n{'='*70}")
    print("Summary:")
    for name, r in all_results.items():
        print(f"  {name:<12}: FFG={r['ffg']:5.0f} FHR={r['fhr']*100:5.0f}% BestFit={r['best_fitness']:.3f}")
    print(f"{'='*70}")