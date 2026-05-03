#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1 — Scaling Experiment
目的: 测试食物空间规模增长对进化算法性能的影响
设计: 7 scales × 7 algorithms × 30 targets × 10 repeats = 14,700 runs
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 1. 数据结构定义
# ============================================================

@dataclass
class FoodSpace:
    """统一食物空间接口"""
    nutrients: np.ndarray  # (n_foods, n_nutrients) float32
    meal_tags: np.ndarray  # (n_foods,) int8, 0/1/2 or -1
    categories: np.ndarray  # (n_foods,) int32
    food_ids: np.ndarray  # (n_foods,) int32

    @property
    def n_foods(self):
        return len(self.food_ids)

    @property
    def n_nutrients(self):
        return self.nutrients.shape[1]


@dataclass
class Target:
    """统一目标接口"""
    target_id: str
    lower_bounds: np.ndarray  # (n_nutrients,)
    upper_bounds: np.ndarray  # (n_nutrients,)
    nutrient_names: List[str]

    @property
    def n_dims(self):
        return len(self.nutrient_names)


@dataclass
class ExperimentResult:
    """单次实验运行结果"""
    experiment: str
    scale: int
    dim: int
    sparsity: float
    algorithm: str
    target_id: str
    rep: int

    FHR: float
    BestFit: float
    FSC: int
    Diversity: float
    Contrib: float
    CV: float
    Entropy: float

    FFG: int
    n_evals: int
    n_foods: float
    runtime: float


# ============================================================
# 2. 数据加载
# ============================================================

def load_nhanes_full() -> FoodSpace:
    """加载 NHANES 全量数据"""
    import common as cm

    nhanes_path = 'data/nhanes.csv'
    food_type_path = 'data/food_type.csv'
    target_path = 'data/target.csv'

    # common.load_data 返回: matrix, meal_tags, nutrients, all_targets
    matrix, meal_tags, nutrients, all_targets = cm.load_data(
        nhanes_path, food_type_path, target_path
    )

    # 所有食物都是有效的（已预过滤）
    all_valid_idx = np.arange(len(matrix), dtype=np.int32)

    nutrients_array = matrix.astype(np.float32)
    meals = meal_tags.astype(np.int8)

    # 计算食物类别
    categories = np.array([int(i) // 1000 if i >= 1000 else i
                           for i in range(len(matrix))], dtype=np.int32)

    return FoodSpace(
        nutrients=nutrients_array,
        meal_tags=meals,
        categories=categories,
        food_ids=all_valid_idx
    )


def sample_food_space(full_space: FoodSpace, n_foods: int, seed: int = 42) -> FoodSpace:
    """从全量空间中子采样"""
    if n_foods >= full_space.n_foods:
        return full_space

    rng = np.random.default_rng(seed)
    indices = rng.choice(full_space.n_foods, size=n_foods, replace=False)
    indices = np.sort(indices)

    return FoodSpace(
        nutrients=full_space.nutrients[indices],
        meal_tags=full_space.meal_tags[indices],
        categories=full_space.categories[indices],
        food_ids=np.arange(n_foods, dtype=np.int32)  # 关键：重新编号为 0..n-1
    )


def generate_synthetic_food_space(n_foods: int, n_nutrients: int, seed: int = 42) -> FoodSpace:
    """生成合成食物空间（用于小规模 baseline）"""
    rng = np.random.default_rng(seed)

    # 营养值服从对数正态分布，模拟真实食物的营养分布
    nutrients = rng.lognormal(mean=3.0, sigma=1.5, size=(n_foods, n_nutrients)).astype(np.float32)

    # 无三餐标记
    meal_tags = np.full(n_foods, -1, dtype=np.int8)

    # 随机分配类别
    n_categories = min(100, n_foods // 10)
    categories = rng.integers(0, n_categories, size=n_foods).astype(np.int32)

    # 生成食物 ID
    food_ids = np.arange(n_foods, dtype=np.int32)

    return FoodSpace(
        nutrients=nutrients,
        meal_tags=meal_tags,
        categories=categories,
        food_ids=food_ids
    )


def load_targets(n_targets: int = 30, seed: int = 42) -> List[Target]:
    """分层抽样 target"""
    import common as cm

    nhanes_path = 'data/nhanes.csv'
    food_type_path = 'data/food_type.csv'
    target_path = 'data/target.csv'

    _, _, nutrients, all_targets = cm.load_data(nhanes_path, food_type_path, target_path)

    rng = np.random.default_rng(seed)

    # 按维度分层
    low_dim = [t for t in all_targets if len(t['nutrients']) <= 6]
    med_dim = [t for t in all_targets if 7 <= len(t['nutrients']) <= 9]
    high_dim = [t for t in all_targets if len(t['nutrients']) >= 10]

    selected = []
    selected.extend(rng.choice(low_dim, size=min(5, len(low_dim)), replace=False))
    selected.extend(rng.choice(med_dim, size=min(20, len(med_dim)), replace=False))
    selected.extend(rng.choice(high_dim, size=min(5, len(high_dim)), replace=False))

    # 补齐到 n_targets
    if len(selected) < n_targets:
        remaining = [t for t in all_targets if t not in selected]
        extra = rng.choice(remaining, size=n_targets - len(selected), replace=False)
        selected.extend(extra)

    targets = []
    for t in selected:
        targets.append(Target(
            target_id=f"target_{t['idx']:04d}",
            lower_bounds=t['target_array'][:, 0],
            upper_bounds=t['target_array'][:, 1],
            nutrient_names=t['nutrients']
        ))

    return targets


# ============================================================
# 3. 算法适配器
# ============================================================

def run_bmoea(food_space: FoodSpace, target: Target, seed: int) -> Dict:
    """运行 BMOEA"""
    from baseline_algorithms import BMOEA
    matrix = food_space.nutrients
    meal_tags = food_space.meal_tags
    all_valid_idx = food_space.food_ids
    max_categories = len(np.unique(food_space.categories))

    target_array = np.column_stack([target.lower_bounds, target.upper_bounds])
    col_idx = list(range(len(target.nutrient_names)))

    np.random.seed(seed)
    algo = BMOEA(matrix, meal_tags, all_valid_idx, max_categories, debug=False)
    result = algo.run(target_array, col_idx)[0]

    return {
        'FHR': result['fhr'],
        'BestFit': result['best_fitness'],
        'FSC': result['fsc'],
        'Diversity': result['diversity'],
        'Contrib': result['contrib'],
        'CV': 1.0 - result['best_fitness'],
        'Entropy': result['diversity'],
        'FFG': result['ffg'],
        'n_evals': result['n_evals'],
        'n_foods': result['avg_n_foods'],
    }


def run_acmo(food_space: FoodSpace, target: Target, seed: int) -> Dict:
    """运行 ACMO-Diet"""
    from baseline_algorithms import ACMODiet
    matrix = food_space.nutrients
    meal_tags = food_space.meal_tags
    all_valid_idx = food_space.food_ids
    max_categories = len(np.unique(food_space.categories))

    target_array = np.column_stack([target.lower_bounds, target.upper_bounds])
    col_idx = list(range(len(target.nutrient_names)))

    np.random.seed(seed)
    algo = ACMODiet(matrix, meal_tags, all_valid_idx, max_categories, debug=False)
    result = algo.run(target_array, col_idx)[0]

    return {
        'FHR': result['fhr'],
        'BestFit': result['best_fitness'],
        'FSC': result['fsc'],
        'Diversity': result['diversity'],
        'Contrib': result['contrib'],
        'CV': 1.0 - result['best_fitness'],
        'Entropy': result['diversity'],
        'FFG': result['ffg'],
        'n_evals': result['n_evals'],
        'n_foods': result['avg_n_foods'],
    }


def run_ma(food_space: FoodSpace, target: Target, seed: int) -> Dict:
    """运行 MA (Memetic Algorithm)"""
    from baseline_algorithms import MemeticAlgorithm
    matrix = food_space.nutrients
    meal_tags = food_space.meal_tags
    all_valid_idx = food_space.food_ids
    max_categories = len(np.unique(food_space.categories))

    target_array = np.column_stack([target.lower_bounds, target.upper_bounds])
    col_idx = list(range(len(target.nutrient_names)))

    np.random.seed(seed)
    algo = MemeticAlgorithm(matrix, meal_tags, all_valid_idx, max_categories, debug=False)
    result = algo.run(target_array, col_idx)[0]

    return {
        'FHR': result['fhr'],
        'BestFit': result['best_fitness'],
        'FSC': result['fsc'],
        'Diversity': result['diversity'],
        'Contrib': result['contrib'],
        'CV': 1.0 - result['best_fitness'],
        'Entropy': result['diversity'],
        'FFG': result['ffg'],
        'n_evals': result['n_evals'],
        'n_foods': result['avg_n_foods'],
    }


def run_pymoo_algo(algo_name: str, food_space: FoodSpace, target: Target, seed: int) -> Dict:
    """运行 pymoo 多目标算法"""
    from baseline_algorithms import run_pymoo_algorithm

    matrix = food_space.nutrients
    meal_tags = food_space.meal_tags
    all_valid_idx = food_space.food_ids
    max_categories = len(np.unique(food_space.categories))

    target_array = np.column_stack([target.lower_bounds, target.upper_bounds])
    col_idx = list(range(len(target.nutrient_names)))

    result = run_pymoo_algorithm(
        algo_name, matrix, target_array, col_idx,
        meal_tags, all_valid_idx, max_categories,
        pop_size=2000, n_generations=50, debug=False
    )

    return {
        'FHR': result['fhr'],
        'BestFit': result['best_fitness'],
        'FSC': result['fsc'],
        'Diversity': result['diversity'],
        'Contrib': result['contrib'],
        'CV': 1.0 - result['best_fitness'],
        'Entropy': result['diversity'],
        'FFG': result['ffg'],
        'n_evals': result['n_evals'],
        'n_foods': result['avg_n_foods'],
    }


# 算法运行映射
ALGO_RUNNERS = {
    'BMOEA': run_bmoea,
    'ACMO-Diet': run_acmo,
    'MA': run_ma,
    'NSGA2': lambda fs, t, s: run_pymoo_algo('NSGA2', fs, t, s),
    'RVEA': lambda fs, t, s: run_pymoo_algo('RVEA', fs, t, s),
    'CTAEA': lambda fs, t, s: run_pymoo_algo('CTAEA', fs, t, s),
    'SPEA2': lambda fs, t, s: run_pymoo_algo('SPEA2', fs, t, s),
}


# ============================================================
# 4. 单次实验运行
# ============================================================

def run_single(
        food_space: FoodSpace,
        target: Target,
        algorithm: str,
        scale: int,
        dim: int,
        sparsity: float,
        rep: int,
        seed: int = 42
) -> ExperimentResult:
    """运行单次实验"""
    run_seed = seed * 10000 + rep

    t_start = time.time()

    try:
        runner = ALGO_RUNNERS[algorithm]
        result = runner(food_space, target, run_seed)
    except Exception as e:
        print(f"  [ERROR] {algorithm} on {target.target_id} rep={rep}: {e}")
        result = {
            'FHR': 0.0, 'BestFit': 0.0, 'FSC': 0,
            'Diversity': 0.0, 'Contrib': 0.0, 'CV': 1.0,
            'Entropy': 0.0, 'FFG': 51, 'n_evals': 0, 'n_foods': 0.0,
        }

    runtime = time.time() - t_start

    return ExperimentResult(
        experiment='scaling',
        scale=scale,
        dim=dim,
        sparsity=sparsity,
        algorithm=algorithm,
        target_id=target.target_id,
        rep=rep,
        FHR=result['FHR'],
        BestFit=result['BestFit'],
        FSC=result['FSC'],
        Diversity=result['Diversity'],
        Contrib=result['Contrib'],
        CV=result['CV'],
        Entropy=result['Entropy'],
        FFG=result['FFG'],
        n_evals=result['n_evals'],
        n_foods=result['n_foods'],
        runtime=runtime
    )


# ============================================================
# 5. 主实验循环
# ============================================================

def run_e1_scaling(config_path: str = 'config.yaml'):
    """E1 主实验"""

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    scales = config['scales']
    algorithms = config['algorithms']
    n_targets = config.get('targets_total', 30)
    n_repeats = config.get('repeats', 10)
    dim = config.get('default_dim', 10)
    sparsity = config.get('default_sparsity', 1.0)
    seed = config.get('seed', 42)
    output_dir = config.get('output_dir', 'results/raw')
    n_workers = config.get('n_workers', 4)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("=" * 70)
    print("E1 — Scaling Experiment")
    print("=" * 70)
    print(f"\n[1/4] Loading NHANES full data...")
    full_space = load_nhanes_full()
    print(f"  Full space: {full_space.n_foods} foods, {full_space.n_nutrients} nutrients")

    print(f"\n[2/4] Loading targets...")
    targets = load_targets(n_targets, seed=seed)
    print(f"  Selected {len(targets)} targets")
    for t in targets[:5]:
        print(f"    {t.target_id}: {t.n_dims} dims, {t.nutrient_names[:5]}...")
    print(f"    ... and {len(targets) - 5} more")

    print(f"\n[3/4] Running experiments...")
    print(f"  Scales: {scales}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Targets: {len(targets)}")
    print(f"  Repeats: {n_repeats}")
    total_runs = len(scales) * len(algorithms) * len(targets) * n_repeats
    print(f"  Total runs: {total_runs}")

    # 准备所有任务
    tasks = []
    for scale in scales:
        # 获取食物空间
        if scale < 210000:
            food_space = sample_food_space(full_space, scale, seed=seed)
        else:
            food_space = full_space

        for target in targets:
            for algorithm in algorithms:
                for rep in range(n_repeats):
                    tasks.append((food_space, target, algorithm, scale, dim, sparsity, rep, seed))

        # 运行任务
    results = []
    completed = 0
    errors = 0
    t_start = time.time()
    last_print_scale = None

    for task in tasks:
        food_space, target, algorithm, scale, dim, sparsity, rep, task_seed = task

        result = run_single(food_space, target, algorithm, scale, dim, sparsity, rep, task_seed)
        results.append(result)
        completed += 1

        if result.FHR == 0.0 and result.BestFit == 0.0:
            errors += 1

        # 进度输出
        if completed % 50 == 0 or completed == 1 or completed == total_runs:
            elapsed = time.time() - t_start
            eta = elapsed / completed * (total_runs - completed) if completed > 0 else 0
            pct = completed / total_runs * 100

            if scale != last_print_scale:
                print(f"\n  --- Scale {scale} ({food_space.n_foods} foods) ---")
                last_print_scale = scale

            print(f"  [{completed}/{total_runs}] {pct:.1f}% | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s | "
                  f"errors={errors} | "
                  f"last: scale={scale} {algorithm} rep={rep} | "
                  f"FHR={result.FHR:.2f} Fit={result.BestFit:.3f} FSC={result.FSC}")

    total_time = time.time() - t_start

    # 保存结果
    print(f"\n[4/4] Saving results...")
    df = pd.DataFrame([asdict(r) for r in results])
    output_path = os.path.join(output_dir, 'e1_scaling.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  {len(df)} rows, {df['algorithm'].nunique()} algorithms")

    # 快速汇总
    print(f"\nQuick Summary (FHR by scale):")
    summary = df.groupby(['scale', 'algorithm'])['FHR'].mean().unstack()
    print(summary.round(3))

    return df


# ============================================================
# 6. 入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E1 Scaling Experiment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    run_e1_scaling(args.config)