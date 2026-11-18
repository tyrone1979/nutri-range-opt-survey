import warnings
import numpy as np
import pandas as pd


warnings.filterwarnings('ignore')
np.random.seed(42)

# ====================== 全局参数 ======================
MIN_FOODS = 3
MAX_FOODS = 30
POP_SIZE = 2000
N_GENERATIONS = 50
MAX_EVALS = 500000
MEAL_COVERAGE_WEIGHT = 100.0
DIVERSITY_WEIGHT = 10.0

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

def get_food_category(food_id):
    food_str = str(int(food_id))
    return food_str[:2] if len(food_str) >= 2 else food_str


def compute_max_categories(all_valid_idx):
    categories = set()
    for fid in all_valid_idx:
        categories.add(get_food_category(fid))
    return len(categories)


def prefilter(matrix, target_array, col_idx):
    max_vals = target_array[:, 1]
    valid = np.all(matrix[:, col_idx] <= max_vals, axis=1)
    return np.where(valid)[0]


def evaluate_batch(population, matrix, target_array, col_idx, meal_tags, max_categories, has_meal_constraint=True):
    pop_size = len(population)
    scores = np.zeros(pop_size, dtype=np.float32)
    feasibles = np.zeros(pop_size, dtype=bool)
    diversities = np.zeros(pop_size, dtype=np.float32)
    n_foods_arr = np.zeros(pop_size, dtype=np.int32)
    contrib_diversities = np.zeros(pop_size, dtype=np.float32)

    for i in range(pop_size):
        ind = population[i]
        n_foods = len(ind)
        n_foods_arr[i] = n_foods
        if n_foods < MIN_FOODS or n_foods > MAX_FOODS:
            continue

        total = matrix[ind][:, col_idx].sum(axis=0)
        lo, hi = target_array[:, 0], target_array[:, 1]
        below = np.maximum(0, lo - total)
        above = np.maximum(0, total - hi)
        nutrient_penalty = float((below + above).sum())
        nutrient_feasible = nutrient_penalty < 1e-6

        # 三餐约束：仅当 has_meal_constraint=True 时检查
        if has_meal_constraint:
            meals = meal_tags[ind]
            meal_coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
            meal_penalty = float((3 - meal_coverage) * MEAL_COVERAGE_WEIGHT)
            meal_feasible = (meal_coverage == 3)
        else:
            meal_penalty = 0.0
            meal_feasible = True

        # 原类别 diversity
        categories = set()
        for fid in ind:
            categories.add(get_food_category(fid))
        diversity_score = len(categories) / max_categories if max_categories > 0 else 0.0
        diversity_penalty = float((1 - diversity_score) * DIVERSITY_WEIGHT)

        # 新增：贡献度 diversity
        contrib_div = compute_contribution_diversity(ind, matrix, col_idx)
        contrib_penalty = (1.0 - contrib_div) * DIVERSITY_WEIGHT * 0.5

        total_penalty = nutrient_penalty + meal_penalty + diversity_penalty + contrib_penalty
        fitness = 1.0 / (1.0 + max(0.0, total_penalty))
        feasible = nutrient_feasible and meal_feasible

        scores[i] = fitness
        feasibles[i] = feasible
        diversities[i] = diversity_score
        contrib_diversities[i] = contrib_div

    return scores, feasibles, diversities, n_foods_arr, contrib_diversities


def random_individual(
    all_valid_idx,
    meal_tags,
    min_foods=MIN_FOODS,
    max_foods=MAX_FOODS,
):
    b_idx = all_valid_idx[meal_tags[all_valid_idx] == 0]
    l_idx = all_valid_idx[meal_tags[all_valid_idx] == 1]
    d_idx = all_valid_idx[meal_tags[all_valid_idx] == 2]

    if len(b_idx) == 0 or len(l_idx) == 0 or len(d_idx) == 0:
        return None

    individual = np.array([
        np.random.choice(b_idx, 1, replace=True)[0],
        np.random.choice(l_idx, 1, replace=True)[0],
        np.random.choice(d_idx, 1, replace=True)[0],
    ], dtype=np.int32)
    lower_extra = max(0, int(min_foods) - len(individual))
    upper_extra = max(lower_extra, min(int(max_foods), len(all_valid_idx)) - len(individual))
    n_extra = np.random.randint(lower_extra, upper_extra + 1)
    if n_extra > 0:
        available = np.setdiff1d(all_valid_idx, individual, assume_unique=False)
        extra = np.random.choice(
            available, min(n_extra, len(available)), replace=False
        )
        individual = np.concatenate([individual, extra])

    return np.unique(individual).astype(np.int32)

import numpy as np


def compute_nutrient_coverage(individual, matrix, col_idx, threshold=0.1):
    """
    向量化版本：营养素覆盖度
    每个营养素至少有一个食物贡献超过阈值
    """
    nutrients = matrix[individual][:, col_idx]  # (n_foods, n_nutrients)
    total = nutrients.sum(axis=0)  # (n_nutrients,)

    # 避免除零
    total_safe = np.where(total > 0, total, 1.0)

    # 每个营养素的最大贡献占比
    max_contrib_ratio = nutrients.max(axis=0) / total_safe

    # 统计超过阈值的营养素数量
    coverage = np.sum((max_contrib_ratio > threshold) & (total > 0))

    return coverage / len(col_idx)


def compute_contribution_entropy(individual, matrix, col_idx):
    """
    向量化版本：贡献熵
    食物对总营养的贡献分布越均匀，熵越高
    """
    nutrients = matrix[individual][:, col_idx]  # (n_foods, n_nutrients)
    food_totals = nutrients.sum(axis=1)  # (n_foods,)
    total_all = food_totals.sum()

    if total_all == 0:
        return 0.0

    probs = food_totals / total_all
    # 向量化计算香农熵
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(individual))

    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_complementarity(individual, matrix, col_idx):
    """
    向量化版本：互补度
    食物两两之间的营养素相关性越低越好

    优化：使用矩阵乘法一次性计算所有两两相关，避免O(n^2) Python循环
    原版本调用np.corrcoef n*(n-1)/2次，新版本只计算1次相关矩阵
    """
    nutrients = matrix[individual][:, col_idx]  # (n_foods, n_nutrients)
    n = len(individual)

    if n < 2:
        return 0.0

    # 标准化每行（每个食物）
    nutrients_centered = nutrients - nutrients.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(nutrients_centered, axis=1, keepdims=True)
    norms_safe = np.where(norms > 0, norms, 1.0)
    nutrients_normed = nutrients_centered / norms_safe

    # 计算相关矩阵: (n,n) = (n,d) @ (d,n)
    corr_matrix = nutrients_normed @ nutrients_normed.T

    # 取上三角（排除对角线），得到所有唯一配对
    triu_indices = np.triu_indices(n, k=1)
    correlations = np.abs(corr_matrix[triu_indices])

    avg_corr = correlations.mean()
    return 1.0 - avg_corr


def compute_dominant_nutrients(individual, matrix, col_idx, dominance_threshold=0.3):
    """
    向量化版本：主导营养素数
    每个食物至少有一个营养素贡献占比 > threshold
    """
    nutrients = matrix[individual][:, col_idx]  # (n_foods, n_nutrients)
    total = nutrients.sum(axis=0)  # (n_nutrients,)

    # 避免除零
    total_safe = np.where(total > 0, total, 1.0)

    # 每个食物对每个营养素的贡献占比
    ratios = nutrients / total_safe  # (n_foods, n_nutrients)

    # 每个食物是否有主导营养素
    has_dominant = np.any(ratios > dominance_threshold, axis=1)

    return has_dominant.sum() / len(individual)


def compute_contribution_diversity(individual, matrix, col_idx):
    """
    向量化版本：综合贡献度
    结合覆盖度、互补度、主导营养素
    """
    coverage = compute_nutrient_coverage(individual, matrix, col_idx)
    entropy = compute_contribution_entropy(individual, matrix, col_idx)
    complementarity = compute_complementarity(individual, matrix, col_idx)
    dominant = compute_dominant_nutrients(individual, matrix, col_idx)

    return 0.3 * coverage + 0.3 * entropy + 0.2 * complementarity + 0.2 * dominant