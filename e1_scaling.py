#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1 — Scaling Experiment (Multi-Dataset)
目的: 测试不同数据集规模对进化算法性能的影响
设计: 7 datasets × 7 algorithms × 12 targets × 10 repeats = 5,880 runs

完整版 — 所有修改已整合
"""

import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据结构定义
# ============================================================
@dataclass
class FoodSpace:
    nutrients: np.ndarray
    meal_tags: np.ndarray
    categories: np.ndarray
    food_ids: np.ndarray
    min_foods: int = 3        # 新增
    max_foods: int = 30       # 新增
    has_meal_constraint: bool = True  # 新增：是否有三餐约束

    @property
    def n_foods(self):
        return len(self.food_ids)

    @property
    def n_nutrients(self):
        return self.nutrients.shape[1]


@dataclass
class Target:
    target_id: str
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    nutrient_names: List[str]
    col_idx: np.ndarray
    target_array: np.ndarray

    @property
    def n_dims(self):
        return len(self.nutrient_names)


@dataclass
class ExperimentResult:
    experiment: str = 'scaling'
    dataset: str = ''
    scale: int = 0
    dim: int = 0
    sparsity: float = 1.0
    algorithm: str = ''
    target_id: str = ''
    rep: int = 0

    FHR: float = 0.0
    BestFit: float = 0.0
    FSC: int = 0
    Diversity: float = 0.0
    Contrib: float = 0.0
    CV: float = 1.0
    Entropy: float = 0.0

    FFG: int = 51
    n_evals: int = 0
    n_foods: float = 0.0
    runtime: float = 0.0


# ============================================================
# 2. 数据加载
# ============================================================

def load_nhanes_full() -> FoodSpace:
    """加载 NHANES 全量数据"""
    import common as cm
    matrix, meal_tags, _, _ = cm.load_data(
        'data/nhanes.csv', 'data/food_type.csv', 'data/target.csv'
    )
    n = len(matrix)
    return FoodSpace(
        nutrients=matrix.astype(np.float32),
        meal_tags=meal_tags.astype(np.int8),
        categories=np.array([i // 1000 if i >= 1000 else i for i in range(n)], dtype=np.int32),
        food_ids=np.arange(n, dtype=np.int32),
        min_foods = 5,  # Nhanes是食物级别
        max_foods = 30,
        has_meal_constraint=True  # 无三餐约束
    )


def sample_food_space(full_space: FoodSpace, n_foods: int, seed: int = 42) -> FoodSpace:
    """
    从 NHANES 分层抽样：按 food_id 前2位（类别）等比例抽取
    保证每个食物类别在子采样中都有代表性
    """
    if n_foods >= full_space.n_foods:
        return full_space

    rng = np.random.default_rng(seed)
    categories = full_space.categories
    unique_cats = np.unique(categories)
    n_total = full_space.n_foods

    sample_indices = []
    for cat in unique_cats:
        cat_idx = np.where(categories == cat)[0]
        # 按该类别在总体中的比例分配配额
        n_sample = max(1, int(n_foods * len(cat_idx) / n_total))
        chosen = rng.choice(cat_idx, size=min(n_sample, len(cat_idx)), replace=False)
        sample_indices.extend(chosen)

    # 如果因取整导致不够 n_foods，从剩余中随机补齐
    if len(sample_indices) < n_foods:
        remaining = list(set(range(n_total)) - set(sample_indices))
        extra = rng.choice(remaining, size=n_foods - len(sample_indices), replace=False)
        sample_indices.extend(extra)

    # 如果超过 n_foods（某些类别配额过多），随机截断
    sample_indices = sample_indices[:n_foods]
    idx = np.array(sorted(sample_indices), dtype=np.int32)

    return FoodSpace(
        nutrients=full_space.nutrients[idx],
        meal_tags=full_space.meal_tags[idx],
        categories=full_space.categories[idx],
        food_ids=np.arange(n_foods, dtype=np.int32),
        min_foods=full_space.min_foods,
        max_foods=full_space.max_foods,
        has_meal_constraint=full_space.has_meal_constraint
    )


def sample_food_space_nested(
    full_space: FoodSpace, n_foods: int, seed: int = 42
) -> FoodSpace:
    """Return a nested, approximately category-stratified prefix.

    Each category receives evenly spaced priority positions on [0, 1), while
    the within-category order is randomized once. Calling this function with
    the same seed and different sizes therefore produces genuinely nested
    candidate sets without discarding category balance.
    """
    if n_foods >= full_space.n_foods:
        return full_space

    rng = np.random.default_rng(seed)
    categories = full_space.categories
    priority_parts = []
    index_parts = []
    for cat in np.unique(categories):
        cat_idx = np.where(categories == cat)[0]
        shuffled = rng.permutation(cat_idx)
        priority = (np.arange(len(shuffled), dtype=float) + 0.5) / len(shuffled)
        priority_parts.append(priority)
        index_parts.append(shuffled)

    priorities = np.concatenate(priority_parts)
    indices = np.concatenate(index_parts)
    order = np.lexsort((indices, priorities))
    idx = np.sort(indices[order[:n_foods]]).astype(np.int32)

    return FoodSpace(
        nutrients=full_space.nutrients[idx],
        meal_tags=full_space.meal_tags[idx],
        categories=full_space.categories[idx],
        food_ids=np.arange(n_foods, dtype=np.int32),
        min_foods=full_space.min_foods,
        max_foods=full_space.max_foods,
        has_meal_constraint=full_space.has_meal_constraint,
    )


def load_allrecipes() -> FoodSpace:
    """加载 Allrecipes (food_data.csv)"""
    df = pd.read_csv('data/food_data.csv')
    cols = ['calories', 'protein', 'carbs', 'fat']
    df = df.dropna(subset=cols)
    for c in cols:
        df = df[df[c] > 0]
    n = len(df)
    return FoodSpace(
        nutrients=df[cols].values.astype(np.float32),
        meal_tags=np.full(n, -1, dtype=np.int8),
        categories=np.zeros(n, dtype=np.int32),
        food_ids=np.arange(n, dtype=np.int32),
        min_foods=3,       # 食谱级别
        max_foods=5,
        has_meal_constraint=False  # 无三餐约束
    )


def load_foods_com() -> FoodSpace:
    """加载 Foods.com (allrecipes.json)"""
    with open('data/allrecipes.json') as f:
        data = json.load(f)
    records = []
    for item in data:
        nut = item.get('nutritions', {})
        if not nut:
            continue
        try:
            records.append({
                'calories': float(nut.get('calories', 0)),
                'protein': float(str(nut.get('protein', '0')).replace('g', '')),
                'carbs': float(str(nut.get('carbs', '0')).replace('g', '')),
                'fat': float(str(nut.get('fat', '0')).replace('g', '')),
            })
        except (ValueError, AttributeError):
            continue
    df = pd.DataFrame(records)
    for c in ['calories', 'protein', 'carbs', 'fat']:
        df = df[df[c] > 0]
    n = len(df)
    return FoodSpace(
        nutrients=df.values.astype(np.float32),
        meal_tags=np.full(n, -1, dtype=np.int8),
        categories=np.zeros(n, dtype=np.int32),
        food_ids=np.arange(n, dtype=np.int32),
        min_foods=3,       # 食谱级别
        max_foods=5,
        has_meal_constraint=False  # 无三餐约束
    )


DATASET_LOADERS = {'allrecipes': load_allrecipes, 'foods_com': load_foods_com}


def generate_small_dataset_targets(food_space, n_targets=6, seed=42):
    """
    为小数据集生成目标范围
    从数据中随机组合三餐，采样营养总和，取 P30-P70 作为范围
    """
    rng = np.random.default_rng(seed)
    matrix = food_space.nutrients
    n_foods = food_space.n_foods
    n_nutrients = food_space.n_nutrients

    # 随机组合膳食并计算营养总和
    samples = []
    for _ in range(1000):
        n_items = rng.integers(food_space.min_foods, food_space.max_foods + 1)
        meal = rng.choice(n_foods, size=n_items, replace=True)
        samples.append(matrix[meal].sum(axis=0))
    samples = np.array(samples)

    targets = []
    for i in range(n_targets):
        rng2 = np.random.default_rng(seed + i * 100)
        # 随机选两个百分位作为范围
        p_low = rng2.uniform(20, 45)
        p_high = rng2.uniform(55, 80)

        lower = np.percentile(samples, p_low, axis=0)
        upper = np.percentile(samples, p_high, axis=0)

        targets.append(Target(
            target_id=f"small_target_{i:02d}",
            lower_bounds=lower.astype(np.float32),
            upper_bounds=upper.astype(np.float32),
            nutrient_names=[f'nutrient_{j}' for j in range(n_nutrients)],
            col_idx=np.arange(n_nutrients, dtype=np.int32),
            target_array=np.column_stack([lower, upper]).astype(np.float32)
        ))

    return targets

def get_food_space(scale_config: dict, full_nhanes: FoodSpace) -> tuple:
    """根据配置获取 (food_space, label, actual_size)"""
    ds = scale_config.get('dataset', 'nhanes_sample')
    size = scale_config.get('size', 0)
    label = scale_config.get('label', str(size))
    if ds in DATASET_LOADERS:
        fs = DATASET_LOADERS[ds]()
    elif ds == 'nhanes_sample':
        fs = sample_food_space(full_nhanes, size)
    elif ds == 'nhanes_full':
        fs = full_nhanes
    else:
        raise ValueError(f"Unknown dataset: {ds}")
    return fs, label, fs.n_foods


def load_targets(n_targets: int = 12, seed: int = 42, exp_type: str = 'scaling', fixed_dim: int = None) -> List[Target]:
    """统一 target 加载"""
    import common as cm
    _, _, _, all_targets = cm.load_data('data/nhanes.csv', 'data/food_type.csv', 'data/target.csv')
    rng = np.random.default_rng(seed)

    # 如果指定了维度，只选该维度的 target
    if fixed_dim is not None:
        valid = [t for t in all_targets if len(t['nutrients']) == fixed_dim]
        print(f"  Filtered to dim={fixed_dim}: {len(valid)} targets")
    else:
        valid = [t for t in all_targets if len(t['nutrients']) >= 6]

    # 按范围宽度评分（越大越宽）
    def score(t):
        r = t['target_array'][:, 1] - t['target_array'][:, 0]
        u = t['target_array'][:, 1]
        return np.mean(r / (u + 1e-6))

    scored = sorted([(t, score(t)) for t in valid], key=lambda x: x[1])
    n = len(scored)

    # 只选最宽的 n_targets 个
    wide = [s[0] for s in scored[2 * n // 3:]]  # 最宽的 1/3
    rng.shuffle(wide)
    selected = wide[:n_targets]

    # 补齐
    if len(selected) < n_targets:
        remaining = [t for t in valid if t not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:n_targets - len(selected)])

    targets = []
    for t in selected[:n_targets]:
        targets.append(Target(
            target_id=f"target_{t['idx']:04d}",
            lower_bounds=t['target_array'][:, 0],
            upper_bounds=t['target_array'][:, 1],
            nutrient_names=t['nutrients'],
            col_idx=t['col_idx'],
            target_array=t['target_array']
        ))

    print(f"  Selected {len(targets)} targets for {exp_type}:")
    for t in targets:
        r = t.upper_bounds - t.lower_bounds
        u = t.upper_bounds
        w = np.mean(r / (u + 1e-6))
        print(f"    {t.target_id}: {t.n_dims}d, width={w:.3f}, {t.nutrient_names}")

    return targets


# ============================================================
# 3. 算法适配器
# ============================================================

def _run_single_algo(algo_cls, food_space, target, seed):
    """通用单目标算法运行"""
    from common import compute_max_categories
    matrix = food_space.nutrients
    meal_tags = food_space.meal_tags
    all_valid_idx = food_space.food_ids
    max_categories = compute_max_categories(all_valid_idx)
    target_array = np.column_stack([target.lower_bounds, target.upper_bounds])
    col_idx = list(range(target.n_dims))
    np.random.seed(seed)
    return algo_cls(matrix, meal_tags, all_valid_idx, max_categories, debug=False).run(target_array, col_idx)[0]

def run_bmoea(fs, t, seed):
    from baseline_algorithms import BMOEA
    from common import compute_max_categories
    target_arr = np.column_stack([t.lower_bounds, t.upper_bounds])
    col_idx = list(range(t.n_dims))
    np.random.seed(seed)
    return BMOEA(fs.nutrients, fs.meal_tags, fs.food_ids,
                 compute_max_categories(fs.food_ids)).run(
        target_arr, col_idx,
        min_foods=fs.min_foods,
        max_foods=fs.max_foods,
        has_meal_constraint=fs.has_meal_constraint
    )[0]

def run_ma(fs, t, seed):
    from baseline_algorithms import MemeticAlgorithm
    from common import compute_max_categories
    target_arr = np.column_stack([t.lower_bounds, t.upper_bounds])
    col_idx = list(range(t.n_dims))
    np.random.seed(seed)
    return MemeticAlgorithm(fs.nutrients, fs.meal_tags, fs.food_ids,
                            compute_max_categories(fs.food_ids)).run(
        target_arr, col_idx,
        min_foods=fs.min_foods,
        max_foods=fs.max_foods,has_meal_constraint=fs.has_meal_constraint
    )[0]

def run_acmo(fs, t, seed):
    from baseline_algorithms import ACMODiet
    from common import compute_max_categories
    target_arr = np.column_stack([t.lower_bounds, t.upper_bounds])
    col_idx = list(range(t.n_dims))
    np.random.seed(seed)
    return ACMODiet(fs.nutrients, fs.meal_tags, fs.food_ids,
                    compute_max_categories(fs.food_ids)).run(
        target_arr, col_idx,
        min_foods=fs.min_foods,
        max_foods=fs.max_foods,has_meal_constraint=fs.has_meal_constraint
    )[0]

def run_pymoo(algo_name, fs, t, seed):
    from baseline_algorithms import run_pymoo_algorithm
    from common import compute_max_categories
    r = run_pymoo_algorithm(
        algo_name, fs.nutrients,
        np.column_stack([t.lower_bounds, t.upper_bounds]),
        list(range(t.n_dims)), fs.meal_tags, fs.food_ids,
        compute_max_categories(fs.food_ids),
        pop_size=2000, n_generations=50, debug=False,
        min_foods=fs.min_foods,
        max_foods=fs.max_foods,
        has_meal_constraint=fs.has_meal_constraint
    )
    return {
        'FHR': r['fhr'], 'BestFit': r['best_fitness'], 'FSC': r['fsc'],
        'Diversity': r['diversity'], 'Contrib': r['contrib'],
        'CV': 1.0 - r['best_fitness'], 'Entropy': r['diversity'],
        'FFG': r['ffg'], 'n_evals': r['n_evals'], 'n_foods': r['avg_n_foods'],
    }

def _extract(result):
    return {
        'FHR': result['fhr'], 'BestFit': result['best_fitness'],
        'FSC': result['fsc'], 'Diversity': result['diversity'],
        'Contrib': result['contrib'], 'CV': 1.0 - result['best_fitness'],
        'Entropy': result['diversity'], 'FFG': result['ffg'],
        'n_evals': result['n_evals'], 'n_foods': result['avg_n_foods'],
    }


ALGO_RUNNERS = {
    'BMOEA': lambda fs, t, s: _extract(run_bmoea(fs, t, s)),
    'ACMO-Diet': lambda fs, t, s: _extract(run_acmo(fs, t, s)),
    'MA': lambda fs, t, s: _extract(run_ma(fs, t, s)),
    'NSGA2': lambda fs, t, s: run_pymoo('NSGA2', fs, t, s),
    'RVEA': lambda fs, t, s: run_pymoo('RVEA', fs, t, s),
    'CTAEA': lambda fs, t, s: run_pymoo('CTAEA', fs, t, s),
    'SPEA2': lambda fs, t, s: run_pymoo('SPEA2', fs, t, s),
}


# ============================================================
# 4. 单次实验运行
# ============================================================

def run_single(food_space, target, algorithm, scale, dim, sparsity, rep, seed=42):
    run_seed = (seed * 10000 + scale + rep) % (2**32 - 1)  # 取模确保合法
    t0 = time.time()
    try:
        res = ALGO_RUNNERS[algorithm](food_space, target, run_seed)
    except Exception as e:
        print(f"  [ERROR] {algorithm} on {target.target_id} rep={rep}: {e}")
        res = {'FHR': 0.0, 'BestFit': 0.0, 'FSC': 0, 'Diversity': 0.0,
               'Contrib': 0.0, 'CV': 1.0, 'Entropy': 0.0,
               'FFG': 51, 'n_evals': 0, 'n_foods': 0.0}
    return ExperimentResult(
        scale=scale, dim=dim, sparsity=sparsity,
        algorithm=algorithm, target_id=target.target_id, rep=rep,
        FHR=res['FHR'], BestFit=res['BestFit'], FSC=res['FSC'],
        Diversity=res['Diversity'], Contrib=res['Contrib'],
        CV=res['CV'], Entropy=res['Entropy'],
        FFG=res['FFG'], n_evals=res['n_evals'],
        n_foods=res['n_foods'], runtime=time.time() - t0
    )


# ============================================================
# 5. 主实验
# ============================================================

def run_e1_scaling(config_path: str = 'config.yaml'):
    """E1 主实验 — 多数据集 Scaling"""

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    scale_configs = config['scales']
    algorithms = config['algorithms']
    n_targets = config.get('targets_total', 12)
    n_repeats = config.get('repeats', 10)
    seed = config.get('seed', 42)
    output_dir = config.get('output_dir', 'results/raw')
    exp_type = config.get('exp_type', 'scaling')

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # [1/4] 加载数据
    # ============================================================
    print("=" * 70)
    print("E1 — Scaling Experiment (Multi-Dataset)")
    print("=" * 70)
    print(f"\n[1/4] Loading datasets...")

    # 加载 NHANES 全量（用于子采样）
    full_nhanes = load_nhanes_full()
    print(f"  NHANES: {full_nhanes.n_foods} foods, {full_nhanes.n_nutrients} nutrients")

    # 预加载 small-scale 数据集
    for sc in scale_configs:
        dataset = sc.get('dataset', 'unknown')
        if dataset in DATASET_LOADERS:
            fs = DATASET_LOADERS[dataset]()
            print(f"  {sc.get('label', dataset)}: {fs.n_foods} foods, {fs.n_nutrients} nutrients")

    # ============================================================
    # [2/4] 加载 targets
    # ============================================================
    print(f"\n[2/4] Loading targets...")

    # NHANES targets（用于 NHANES 及子采样）
    nhanes_targets = load_targets(n_targets, seed=seed, exp_type=exp_type, fixed_dim=6)
    print(f"  NHANES targets: {len(nhanes_targets)}")
    for t in nhanes_targets[:3]:
        print(f"    {t.target_id}: {t.n_dims}d, {t.nutrient_names[:4]}...")

    # 小数据集 targets（从数据本身生成）
    small_targets = {}
    for sc in scale_configs:
        ds = sc.get('dataset', '')
        if ds in DATASET_LOADERS:
            fs = DATASET_LOADERS[ds]()
            small_targets[ds] = generate_small_dataset_targets(
                fs, n_targets=6, seed=seed
            )
            print(f"  {ds} targets: {len(small_targets[ds])}")
            for t in small_targets[ds][:2]:
                print(f"    {t.target_id}: {t.n_dims}d, ranges={list(zip(t.lower_bounds, t.upper_bounds))}")

    # ============================================================
    # [3/4] 运行实验
    # ============================================================
    # 计算总运行量
    total = 0
    for sc in scale_configs:
        ds = sc.get('dataset', '')
        n_tgt = len(small_targets.get(ds, nhanes_targets))
        total += len(algorithms) * n_tgt * n_repeats

    print(f"\n[3/4] Running {total} experiments...")
    print(f"  Algorithms: {algorithms}")
    print(f"  Repeats: {n_repeats}")

    results, n_done, n_err, t0 = [], 0, 0, time.time()
    last_label = None

    for sc in scale_configs:
        fs, label, actual_size = get_food_space(sc, full_nhanes)
        ds_name = sc.get('dataset', '')

        # 选择 targets
        if ds_name in small_targets:
            targets = small_targets[ds_name]
        else:
            targets = nhanes_targets

        print(f"\n  --- {label} ({actual_size} foods, {len(targets)} targets) ---")

        for tgt in targets:
            # 维度对齐
            max_dim = fs.n_nutrients
            if ds_name in small_targets:
                at = tgt  # 已匹配
            elif tgt.n_dims > max_dim:
                at = Target(
                    tgt.target_id,
                    tgt.lower_bounds[:max_dim],
                    tgt.upper_bounds[:max_dim],
                    tgt.nutrient_names[:max_dim],
                    tgt.col_idx[:max_dim],
                    tgt.target_array[:max_dim]
                )
            else:
                at = tgt

            for alg in algorithms:
                for rep in range(n_repeats):
                    task_seed = seed * 10000 + actual_size + rep

                    r = run_single(fs, at, alg, actual_size, at.n_dims, 1.0, rep, task_seed)
                    r.dataset = ds_name
                    results.append(r)
                    n_done += 1

                    if r.FHR == 0.0 and r.BestFit == 0.0:
                        n_err += 1

                    # 进度输出
                    if n_done % 50 == 0 or n_done == 1 or n_done == total:
                        el = time.time() - t0
                        eta = el / n_done * (total - n_done) if n_done > 0 else 0
                        pct = n_done / total * 100

                        if label != last_label:
                            last_label = label

                        print(f"  [{n_done}/{total}] {pct:.1f}% | "
                              f"elapsed={el:.0f}s | ETA={eta:.0f}s | "
                              f"errors={n_err} | "
                              f"{label} {alg} r{rep} | "
                              f"FHR={r.FHR:.2f} Fit={r.BestFit:.3f} FSC={r.FSC}")

    total_time = time.time() - t0

    # ============================================================
    # [4/4] 保存结果
    # ============================================================
    print(f"\n[4/4] Saving results...")
    print(f"  Completed: {n_done}/{total}")
    print(f"  Errors: {n_err}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    df = pd.DataFrame([asdict(r) for r in results])
    output_path = os.path.join(output_dir, 'e1_scaling.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  {len(df)} rows")

    # ============================================================
    # 快速汇总
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Quick Summary: FHR by Dataset and Algorithm")
    print(f"{'=' * 70}")

    try:
        summary = df.groupby(['dataset', 'scale', 'algorithm'])['FHR'].mean().unstack()
        print(summary.round(3))
    except Exception as e:
        print(f"  Summary error: {e}")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    run_e1_scaling(args.config)