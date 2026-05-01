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


def evaluate_batch(population, matrix, target_array, col_idx, meal_tags, max_categories):
    pop_size = len(population)
    scores = np.zeros(pop_size, dtype=np.float32)
    feasibles = np.zeros(pop_size, dtype=bool)
    diversities = np.zeros(pop_size, dtype=np.float32)
    n_foods_arr = np.zeros(pop_size, dtype=np.int32)  # 新增

    for i in range(pop_size):
        ind = population[i]
        n_foods = len(ind)
        n_foods_arr[i] = n_foods  # 记录
        if n_foods < MIN_FOODS or n_foods > MAX_FOODS:
            continue

        total = matrix[ind][:, col_idx].sum(axis=0)
        lo, hi = target_array[:, 0], target_array[:, 1]
        below = np.maximum(0, lo - total)
        above = np.maximum(0, total - hi)
        nutrient_penalty = float((below + above).sum())
        nutrient_feasible = nutrient_penalty < 1e-6

        meals = meal_tags[ind]
        meal_coverage = int(np.any(meals == 0)) + int(np.any(meals == 1)) + int(np.any(meals == 2))
        meal_penalty = float((3 - meal_coverage) * MEAL_COVERAGE_WEIGHT)

        categories = set()
        for fid in ind:
            categories.add(get_food_category(fid))
        diversity_score = len(categories) / max_categories if max_categories > 0 else 0.0
        diversity_penalty = float((1 - diversity_score) * DIVERSITY_WEIGHT)

        total_penalty = nutrient_penalty + meal_penalty + diversity_penalty
        fitness = 1.0 / (1.0 + max(0.0, total_penalty))
        feasible = nutrient_feasible and (meal_coverage == 3)

        scores[i] = fitness
        feasibles[i] = feasible
        diversities[i] = diversity_score

    return scores, feasibles, diversities, n_foods_arr  # 返回4个


def random_individual(all_valid_idx, meal_tags):
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

