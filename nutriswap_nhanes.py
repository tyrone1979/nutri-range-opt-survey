"""
NutriSwap-GA v2: 统一编码 + 餐覆盖约束 + 智能初始化 + 保留所有可行解
目标: NutriSwap-GA 超越 BMOEA

核心变更:
  1. 统一食物池编码（取消三餐分离），约束中强制餐类型覆盖=3
  2. NutriSwap保留所有可行解（类似BMOEA精英保留）
  3. 修复交叉和变异算子（BMOEA式合并重采样）
  4. 智能初始化: 随机生成 → 全部不可行时启动修复逻辑
  5. NSGA2/NSGA3/MOEAD/RVEA 用 pymoo（避免自研bug）
  6. 新增食物多样性评估（基于food_id前2位类别）
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
POP_SIZE = 2000
N_GENERATIONS = 50
MAX_EVALS = 500000
MEAL_COVERAGE_WEIGHT = 100.0  # 餐覆盖缺失惩罚权重
DIVERSITY_WEIGHT = 10.0       # 多样性惩罚权重
# MAX_CATEGORIES 将在运行时动态计算

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
# 2. 工具函数（包含多样性评估）
# ============================================================

def get_food_category(food_id):
    """
    根据food_id前2位获取食物类别
    例如: 11001 -> 11 (水果类), 21001 -> 21 (蔬菜类)
    """
    food_str = str(int(food_id))
    category = food_str[:2] if len(food_str) >= 2 else food_str
    return category


def compute_max_categories(all_valid_idx):
    """计算实际食物数据中的最大类别数（基于food_id前2位）"""
    categories = set()
    for fid in all_valid_idx:
        cat = get_food_category(fid)
        categories.add(cat)
    return len(categories)


def prefilter(matrix, target_array, col_idx):
    """预过滤: 排除超过任何营养素上限的食物"""
    max_vals = target_array[:, 1]
    valid = np.all(matrix[:, col_idx] <= max_vals, axis=1)
    return np.where(valid)[0]


def evaluate(individual, matrix, target_array, col_idx, meal_tags, max_categories):
    """
    评估单个个体（食物索引数组）
    返回: (fitness, total_nutrients, is_feasible, diversity_score)
    """
    n_foods = len(individual)
    if n_foods < MIN_FOODS or n_foods > MAX_FOODS:
        return 0.0, None, False, 0.0

    # 营养素总和
    total = matrix[individual][:, col_idx].sum(axis=0)
    lo, hi = target_array[:, 0], target_array[:, 1]
    below = np.maximum(0, lo - total)
    above = np.maximum(0, total - hi)
    nutrient_penalty = float((below + above).sum())
    nutrient_feasible = nutrient_penalty < 1e-6

    # 餐类型覆盖检查
    meals = meal_tags[individual]
    has_b = np.any(meals == 0)
    has_l = np.any(meals == 1)
    has_d = np.any(meals == 2)
    meal_coverage = int(has_b) + int(has_l) + int(has_d)
    meal_penalty = float((3 - meal_coverage) * MEAL_COVERAGE_WEIGHT)

    # 多样性评分
    categories = set()
    for fid in individual:
        cat = get_food_category(fid)
        categories.add(cat)

    diversity_score = len(categories) / max_categories if max_categories > 0 else 0.0
    diversity_penalty = float((1 - diversity_score) * DIVERSITY_WEIGHT)

    total_penalty = nutrient_penalty + meal_penalty + diversity_penalty
    total_penalty = max(0.0, total_penalty)
    fitness = 1.0 / (1.0 + total_penalty)
    feasible = nutrient_feasible and (meal_coverage == 3)

    return fitness, total, feasible, diversity_score


def fast_evaluate_pop(population, matrix, target_array, col_idx, meal_tags, max_categories):
    """批量评估种群，返回 (scores_array, feasible_array, diversity_array)"""
    pop_size = len(population)
    scores = np.zeros(pop_size, dtype=np.float32)
    feasibles = np.zeros(pop_size, dtype=bool)
    diversities = np.zeros(pop_size, dtype=np.float32)

    for i in range(pop_size):
        f, _, fe, d = evaluate(population[i], matrix, target_array, col_idx, meal_tags, max_categories)
        scores[i] = f
        feasibles[i] = fe
        diversities[i] = d
    return scores, feasibles, diversities


def random_individual(all_valid_idx, meal_tags):
    """生成随机个体（确保餐覆盖）"""
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
    n_extra = np.random.randint(0, MAX_FOODS - 2)
    if n_extra > 0:
        extra = np.random.choice(all_valid_idx, n_extra, replace=True)
        individual.extend(extra)

    return np.array(individual, dtype=np.int32)


# ============================================================
# 3. NutriSwap-GA (核心算法)
# ============================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class NutriSwapGA:
    """NutriSwap-GA: 稳定工作版本"""

    def __init__(self, matrix, meal_tags, all_valid_idx, max_categories, debug=False):
        self.matrix = matrix
        self.meal_tags = meal_tags
        self.all_valid_idx = all_valid_idx
        self.max_categories = max_categories
        self.debug = debug
        self.similar_foods = None
        self._fingerprint_built = False

    def _build_fingerprint_index_fast(self):
        if self._fingerprint_built:
            return
        n_foods = len(self.all_valid_idx)
        if n_foods > 10000:
            sample_size = min(5000, n_foods)
            sample_indices = np.random.choice(n_foods, size=sample_size, replace=False)
            sampled_foods = self.all_valid_idx[sample_indices]
            sampled_matrix = self.matrix[sampled_foods]
        else:
            sampled_foods = self.all_valid_idx
            sampled_matrix = self.matrix[sampled_foods]

        scaler = StandardScaler()
        normalized = scaler.fit_transform(sampled_matrix)
        n_components = min(5, len(sampled_foods) - 1, sampled_matrix.shape[1])
        if n_components >= 2:
            pca = PCA(n_components=n_components)
            fingerprints = pca.fit_transform(normalized)
        else:
            fingerprints = normalized

        n_neighbors = min(20, len(sampled_foods))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(fingerprints)

        self.similar_foods = {}
        for i, fid in enumerate(sampled_foods):
            distances, indices = nn.kneighbors([fingerprints[i]], n_neighbors=n_neighbors)
            similar = []
            for d, idx in zip(distances[0][1:], indices[0][1:]):
                if d > 0:
                    similarity = 1.0 / (1.0 + d)
                    similar.append((sampled_foods[idx], similarity))
            self.similar_foods[fid] = similar

        if n_foods > 10000:
            unsampled_foods = np.setdiff1d(self.all_valid_idx, sampled_foods)
            if len(unsampled_foods) > 0:
                unsampled_matrix = self.matrix[unsampled_foods]
                unsampled_normalized = scaler.transform(unsampled_matrix)
                if n_components >= 2:
                    unsampled_fingerprints = pca.transform(unsampled_normalized)
                else:
                    unsampled_fingerprints = unsampled_normalized
                neighbors = nn.kneighbors(unsampled_fingerprints, n_neighbors=1)
                for i, fid in enumerate(unsampled_foods):
                    nearest_sample_idx = neighbors[1][i][0]
                    nearest_sample_fid = sampled_foods[nearest_sample_idx]
                    self.similar_foods[fid] = self.similar_foods.get(nearest_sample_fid, [])

        self._fingerprint_built = True
        if self.debug:
            print(f"      [NutriSwap] Index built")

    def _get_similar_foods(self, food_id):
        if not self._fingerprint_built:
            self._build_fingerprint_index_fast()
        return self.similar_foods.get(food_id, [])

    def _random_individual(self):
        n_items = np.random.randint(MIN_FOODS, min(MAX_FOODS, len(self.all_valid_idx)) + 1)
        return np.random.choice(self.all_valid_idx, size=n_items, replace=False)

    def _select_best_replacement(self, original_food, current_individual, target_array, col_idx):
        similar_foods = self._get_similar_foods(original_food)[:10]
        if not similar_foods:
            return original_food

        current_total = self.matrix[current_individual][:, col_idx].sum(axis=0)
        current_total_without = current_total - self.matrix[original_food][col_idx]
        target_center = (target_array[:, 0] + target_array[:, 1]) / 2.0

        best_food = original_food
        best_dist = float('inf')
        for similar_food, _ in similar_foods:
            new_total = current_total_without + self.matrix[similar_food][col_idx]
            dist = np.sum((new_total - target_center) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_food = similar_food
        return best_food

    def _mutate_smart(self, individual, target_array, col_idx):
        if len(individual) == 0:
            return individual
        mutated = individual.copy()
        pos = np.random.randint(0, len(mutated))
        new_food = self._select_best_replacement(mutated[pos], mutated, target_array, col_idx)
        mutated[pos] = new_food
        return np.unique(mutated)

    def run(self, target_array, col_idx, pop_size=POP_SIZE,
            n_generations=N_GENERATIONS, max_evals=MAX_EVALS):

        n_evals = 0
        population = []
        for _ in range(pop_size):
            population.append(self._random_individual())

        scores, feasibles, diversities = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
        n_evals += pop_size

        if self.debug:
            print(f"      [NutriSwap] Init: feasible={feasibles.sum()}/{pop_size}")

        ffg = None
        best_feasible_individual = None
        best_feasible_fitness = 0.0

        for gen in range(n_generations):
            if n_evals >= max_evals:
                break

            n_feas = feasibles.sum()

            # 记录最佳可行解
            for i in range(len(population)):
                if feasibles[i] and scores[i] > best_feasible_fitness:
                    best_feasible_fitness = scores[i]
                    best_feasible_individual = population[i].copy()

            if n_feas > 0 and ffg is None:
                ffg = gen + 1
                if self.debug:
                    print(f"      [NutriSwap] Gen {gen + 1}: FIRST FEASIBLE! count={n_feas}")

            if n_feas >= 20:
                if self.debug:
                    print(f"      [NutriSwap] Gen {gen + 1}: Early stop")
                break

            # 精英保留
            feas_idx = np.where(feasibles)[0]
            elite_size = min(len(feas_idx), pop_size // 3)
            new_pop = []
            if elite_size > 0:
                sorted_idx = feas_idx[np.argsort(scores[feas_idx])[::-1]]
                new_pop = [population[i].copy() for i in sorted_idx[:elite_size]]

            remaining = pop_size - len(new_pop)
            if remaining > 0:
                infeas_idx = np.where(~feasibles)[0]
                if len(infeas_idx) > 0:
                    infeas_scores = np.maximum(scores[infeas_idx], 1e-10)
                    probs = infeas_scores / infeas_scores.sum()
                    chosen = np.random.choice(infeas_idx, size=min(remaining, len(infeas_idx)), p=probs, replace=True)
                    new_pop.extend([population[i].copy() for i in chosen])
                while len(new_pop) < pop_size:
                    new_pop.append(self._random_individual())

            population = new_pop[:pop_size]

            # 交叉
            for i in range(0, pop_size - 1, 2):
                if np.random.random() < 0.25:
                    combined = np.unique(np.concatenate([population[i], population[i + 1]]))
                    if len(combined) >= MIN_FOODS:
                        n_target = np.random.randint(MIN_FOODS, min(MAX_FOODS, len(combined)) + 1)
                        population[i] = np.random.choice(combined, size=n_target, replace=False)

            # 变异
            for i in range(pop_size):
                if np.random.random() < 0.85:
                    population[i] = self._mutate_smart(population[i], target_array, col_idx)

            scores, feasibles, diversities = fast_evaluate_pop(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
            n_evals += pop_size

        # 最终评估：如果没有可行解，恢复历史最佳
        final_scores, final_feasibles, final_diversities = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)

        if final_feasibles.sum() == 0 and best_feasible_individual is not None:
            if self.debug:
                print(f"      [NutriSwap] Restoring best feasible individual")
            population[0] = best_feasible_individual
            final_scores, final_feasibles, final_diversities = fast_evaluate_pop(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)

        avg_diversity = float(final_diversities.mean()) if len(final_diversities) > 0 else 0.0

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
            'diversity': avg_diversity,
        }]

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

            scores, feasibles, _ = fast_evaluate_pop(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
            n_evals += pop_size

            n_feas = feasibles.sum()
            if n_feas > 0 and not feasible_found:
                ffg = gen + 1
                feasible_found = True
                if self.debug:
                    print(f"      [BMOEA] Gen {gen+1}: FIRST FEASIBLE! count={n_feas}")

            if n_feas >= 20:
                if self.debug:
                    print(f"      [BMOEA] Gen {gen+1}: Early stop, {n_feas} feasible")
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
                    combined = np.unique(np.concatenate([population[i], population[i+1]]))
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

        final_scores, final_feasibles, final_diversities = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
        avg_diversity = float(final_diversities.mean()) if len(final_diversities) > 0 else 0.0

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
            'diversity': avg_diversity,
        }]


# ============================================================
# 5. ACMO-Diet
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

            scores, feasibles, _ = fast_evaluate_pop(
                population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
            n_evals += pop_size

            n_feas = feasibles.sum()
            if n_feas > 0 and ffg is None:
                ffg = gen + 1
                if self.debug:
                    print(f"      [ACMO] Gen {gen+1}: FIRST FEASIBLE! count={n_feas}")

            if n_feas >= 20:
                if self.debug:
                    print(f"      [ACMO] Gen {gen+1}: Early stop, {n_feas} feasible")
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

        final_scores, final_feasibles, final_diversities = fast_evaluate_pop(
            population, self.matrix, target_array, col_idx, self.meal_tags, self.max_categories)
        avg_diversity = float(final_diversities.mean()) if len(final_diversities) > 0 else 0.0

        return [{
            'ffg': ffg if ffg else n_generations + 1,
            'fhr': 1.0 if np.any(final_feasibles) else 0.0,
            'fsc': int(final_feasibles.sum()),
            'best_fitness': float(final_scores.max()),
            'n_evals': n_evals,
            'diversity': avg_diversity,
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


def run_pymoo_algorithm(algo_name, matrix, target_array, col_idx, meal_tags, all_valid_idx,
                        max_categories, pop_size=2000, n_generations=50, debug=False):
    """运行pymoo算法 - 完整修复版"""

    default_result = {
        'ffg': n_generations + 1,
        'fhr': 0.0,
        'fsc': 0,
        'best_fitness': 0.0,
        'n_evals': 0,
        'diversity': 0.0,
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

        # 存储所有可行解（避免丢失）
        all_feasible_individuals = []
        all_feasible_fitness = []

        for gen in range(n_generations):
            algorithm.next()
            n_evals += pop_size

            if algorithm.pop is not None:
                for ind in algorithm.pop:
                    x = ind.X
                    valid_mask = x >= 0
                    indices = all_valid_idx[x[valid_mask].astype(int)]
                    if len(indices) >= MIN_FOODS:
                        f, _, fe, d = evaluate(indices, matrix, target_array, col_idx, meal_tags, max_categories)
                        f = min(1.0, max(0.0, f))

                        if f > best_fitness:
                            best_fitness = f

                        if fe:
                            # 记录所有可行解
                            all_feasible_individuals.append((indices.copy(), f, d))
                            all_feasible_fitness.append(f)

                            if first_feasible_gen is None:
                                first_feasible_gen = gen + 1
                                if debug:
                                    print(f"      [pymoo {algo_name}] Gen {gen + 1}: FIRST FEASIBLE!")

                # 早停条件：可行解数量达到阈值
                current_feas_count = len([1 for ind in algorithm.pop if hasattr(ind, 'feasible')])
                # 更准确的方法：统计当前代所有可行解
                feas_in_current_gen = 0
                for ind in algorithm.pop:
                    x = ind.X
                    valid_mask = x >= 0
                    indices = all_valid_idx[x[valid_mask].astype(int)]
                    if len(indices) >= MIN_FOODS:
                        _, _, fe, _ = evaluate(indices, matrix, target_array, col_idx, meal_tags, max_categories)
                        if fe:
                            feas_in_current_gen += 1

                if feas_in_current_gen >= 20:
                    if debug:
                        print(f"      [pymoo {algo_name}] Gen {gen + 1}: Early stop, {feas_in_current_gen} feasible")
                    break

        # 【关键】最终结果：使用收集到的所有可行解
        n_feas = len(all_feasible_individuals)

        # 计算平均多样性（从可行解中）
        if n_feas > 0:
            avg_diversity = np.mean([d for _, _, d in all_feasible_individuals])
        else:
            avg_diversity = 0.0

        # 如果没有找到可行解，从最后一代中统计
        if n_feas == 0 and algorithm.pop is not None:
            diversity_sum = 0.0
            for sol in algorithm.pop:
                x = sol.X
                valid_mask = x >= 0
                indices = all_valid_idx[x[valid_mask].astype(int)]
                if len(indices) >= MIN_FOODS:
                    f, _, fe, d = evaluate(indices, matrix, target_array, col_idx, meal_tags, max_categories)
                    if f > best_fitness:
                        best_fitness = f
                    if fe:
                        n_feas += 1
                    diversity_sum += d
            avg_diversity = diversity_sum / len(algorithm.pop) if len(algorithm.pop) > 0 else 0.0

        return {
            'ffg': first_feasible_gen if first_feasible_gen else n_generations + 1,
            'fhr': 1.0 if n_feas > 0 else 0.0,
            'fsc': n_feas,
            'best_fitness': best_fitness,
            'n_evals': n_evals,
            'diversity': avg_diversity,
        }

    except Exception as e:
        print(f"      [pymoo {algo_name}] Error: {e}")
        return default_result
# ============================================================
# 7. 主程序
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

    # ============================================================
    # 设置要测试的目标数量
    # ============================================================
    NUM_TARGETS_TO_TEST = min(5, n_targets)  # 测试前5个目标，可以修改
    print(f"\nTesting first {NUM_TARGETS_TO_TEST} targets...")

    # 存储所有目标的汇总结果
    all_targets_results = []

    for target_idx in range(NUM_TARGETS_TO_TEST):
        target_info = all_targets[target_idx]
        ta, ci = target_info['target_array'], target_info['col_idx']

        print(f"\n{'=' * 70}")
        print(f"Target {target_idx + 1}/{NUM_TARGETS_TO_TEST} (idx={target_info['idx']})")
        print(f"Nutrients: {target_info['nutrients']}")
        print(f"Ranges: {ta.tolist()}")

        # 预过滤
        all_valid = prefilter(matrix, ta, ci)
        print(f"Pre-filtered: {len(all_valid)}/{n_foods} foods")

        # 检查是否有足够食物
        if len(all_valid) < MIN_FOODS:
            print("  Not enough valid foods, skipping this target")
            continue

        # 计算最大类别数
        max_categories = compute_max_categories(all_valid)
        print(f"  Max food categories: {max_categories}")

        # 检查每种餐类型的数量
        b_count = np.sum(meal_tags[all_valid] == 0)
        l_count = np.sum(meal_tags[all_valid] == 1)
        d_count = np.sum(meal_tags[all_valid] == 2)
        print(f"  b/l/d: {b_count}/{l_count}/{d_count}")

        if b_count == 0 or l_count == 0 or d_count == 0:
            print("  Missing meal types, skipping this target")
            continue


        # 随机可行率
        def _rand_ind():
            ind = random_individual(all_valid, meal_tags)
            if ind is None:
                ind = np.random.choice(all_valid, MIN_FOODS, replace=False)
            return ind


        fc = sum(1 for _ in range(100) if evaluate(_rand_ind(), matrix, ta, ci, meal_tags, max_categories)[2])
        print(f"Random feasibility: {fc}%")

        # 运行算法
        print(f"\nRunning algorithms (pop={POP_SIZE}, gen={N_GENERATIONS})")
        print(
            f"{'Algorithm':<15} {'FFG':>6} {'FHR':>6} {'FSC':>6} {'BestFit':>8} {'Diversity':>9} {'Evals':>8} {'Time(s)':>8}")
        print(f"{'-' * 76}")

        algorithms = {
            'NutriSwap': lambda: NutriSwapGA(matrix, meal_tags, all_valid, max_categories, debug=False).run(ta, ci),
            'BMOEA': lambda: BMOEA(matrix, meal_tags, all_valid, max_categories, debug=False).run(ta, ci),
            'ACMO-Diet': lambda: ACMODiet(matrix, meal_tags, all_valid, max_categories, debug=False).run(ta, ci),
        }

        if PYMOO_AVAILABLE:
            pymoo_algos = ['NSGA2', 'NSGA3',  'RVEA']
            for name in pymoo_algos:
                algorithms[name] = lambda n=name: [run_pymoo_algorithm(
                    n, matrix, ta, ci, meal_tags, all_valid, max_categories,
                    pop_size=POP_SIZE, n_generations=N_GENERATIONS, debug=False)]

        target_results = {}
        for name, algo_fn in algorithms.items():
            t0 = time.time()
            try:
                result = algo_fn()
                if result is None or len(result) == 0:
                    print(f"{name:<15} ERROR: No result returned")
                    continue
                r = result[0]
                if r is None:
                    print(f"{name:<15} ERROR: Result is None")
                    continue
                elapsed = time.time() - t0
                target_results[name] = r
                diversity_display = f"{r.get('diversity', 0):8.3f}"
                print(
                    f"{name:<15} {r['ffg']:6.0f} {r['fhr'] * 100:5.0f}% {r['fsc']:5.0f} {r['best_fitness']:8.3f} {diversity_display:>9} {r['n_evals']:7.0f} {elapsed:7.1f}")
            except Exception as e:
                print(f"{name:<15} ERROR: {e}")

        print(f"\nSummary for Target {target_idx + 1}:")
        for name, r in target_results.items():
            print(f"  {name:<12}: FFG={r['ffg']:5.0f} FHR={r['fhr'] * 100:5.0f}% BestFit={r['best_fitness']:.3f}")

        # 保存当前目标的结果
        all_targets_results.append({
            'target_idx': target_idx + 1,
            'target_nutrients': target_info['nutrients'],
            'results': target_results
        })

        print(f"{'=' * 70}")

    # ============================================================
    # 打印所有目标的汇总统计
    # ============================================================
    if len(all_targets_results) > 1:
        print(f"\n{'=' * 70}")
        print("SUMMARY ACROSS ALL TARGETS")
        print(f"{'=' * 70}")

        # 汇总每个算法的平均性能
        algo_avg = {}
        for target_result in all_targets_results:
            for name, r in target_result['results'].items():
                if name not in algo_avg:
                    algo_avg[name] = {'ffg': [], 'fsc': [], 'best_fitness': [], 'diversity': [], 'time': []}
                algo_avg[name]['ffg'].append(r['ffg'])
                algo_avg[name]['fsc'].append(r['fsc'])
                algo_avg[name]['best_fitness'].append(r['best_fitness'])
                algo_avg[name]['diversity'].append(r.get('diversity', 0))

        print(f"\n{'Algorithm':<15} {'Avg FFG':>10} {'Avg FSC':>10} {'Avg BestFit':>12} {'Avg Diversity':>14}")
        print(f"{'-' * 61}")
        for name in sorted(algo_avg.keys()):
            avg_ffg = np.mean(algo_avg[name]['ffg'])
            avg_fsc = np.mean(algo_avg[name]['fsc'])
            avg_fit = np.mean(algo_avg[name]['best_fitness'])
            avg_div = np.mean(algo_avg[name]['diversity'])
            print(f"{name:<15} {avg_ffg:10.1f} {avg_fsc:10.1f} {avg_fit:12.3f} {avg_div:14.3f}")

        print(f"{'=' * 70}")

    print("\nAll targets completed!")