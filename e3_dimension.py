#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3 — Dimension Experiment
目的: 测试营养素维度增长对算法性能的影响
设计: 4 dimensions × 7 algorithms × 12 targets × 10 repeats = 3,360 runs

重用 E1 的所有组件
"""

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from dataclasses import asdict
import warnings

warnings.filterwarnings('ignore')

# 从 E1 导入所有可重用组件
from e1_scaling import (
    FoodSpace, Target, ExperimentResult,
    load_nhanes_full, load_targets,
    ALGO_RUNNERS, run_single
)


# ============================================================
# E3 特有：维度投影
# ============================================================

def project_food_space(food_space: FoodSpace, dim: int) -> FoodSpace:
    """将食物空间投影到指定维度（取前 dim 个营养素）"""
    return FoodSpace(
        nutrients=food_space.nutrients[:, :dim],
        meal_tags=food_space.meal_tags,
        categories=food_space.categories,
        food_ids=food_space.food_ids
    )


def project_target(target: Target, dim: int) -> Target:
    """将 target 投影到指定维度"""
    # 取前 dim 个营养素
    actual_dim = min(dim, len(target.nutrient_names))

    return Target(
        target_id=target.target_id,
        lower_bounds=target.lower_bounds[:actual_dim],
        upper_bounds=target.upper_bounds[:actual_dim],
        nutrient_names=target.nutrient_names[:actual_dim],
        col_idx=target.col_idx[:actual_dim],
        target_array=target.target_array[:actual_dim]
    )


# ============================================================
# E3 主实验
# ============================================================

def run_e3_dimension(config_path: str = 'config.yaml'):
    """E3 主实验"""

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    algorithms = config['algorithms']
    dimensions = config.get('dimensions', [6, 9, 12, 16])
    n_targets = config.get('targets_total', 12)
    n_repeats = config.get('repeats', 10)
    seed = config.get('seed', 42)
    output_dir = config.get('output_dir', 'results/raw')

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # [1/4] 加载数据
    # ============================================================
    print("=" * 70)
    print("E3 — Dimension Experiment")
    print("=" * 70)
    print(f"\n[1/4] Loading NHANES full data...")

    full_space = load_nhanes_full()
    print(f"  Full space: {full_space.n_foods} foods, {full_space.n_nutrients} nutrients")

    scale = full_space.n_foods
    sparsity = 1.0  # 固定 normal 约束

    # ============================================================
    # [2/4] 加载 targets
    # ============================================================
    print(f"\n[2/4] Loading targets...")
    base_targets = load_targets(n_targets, seed=seed)
    print(f"  Selected {len(base_targets)} targets")
    for t in base_targets[:3]:
        print(f"    {t.target_id}: {t.n_dims} dims, {t.nutrient_names[:3]}...")
    if len(base_targets) > 3:
        print(f"    ... and {len(base_targets) - 3} more")

    # ============================================================
    # [3/4] 运行实验
    # ============================================================
    print(f"\n[3/4] Running experiments...")
    print(f"  Dimensions: {dimensions}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Targets: {len(base_targets)}")
    print(f"  Repeats: {n_repeats}")
    total_runs = len(dimensions) * len(algorithms) * len(base_targets) * n_repeats
    print(f"  Total runs: {total_runs}")

    # 检查是否可以复用 E1 的 dim=16 数据（E1 用的是全量 16 维）
    e1_path = os.path.join(output_dir, 'e1_scaling.csv')
    e1_df = None
    if os.path.exists(e1_path):
        e1_df = pd.read_csv(e1_path)
        e1_full_dim = e1_df[(e1_df['scale'] == 210000) & (e1_df['sparsity'] == 1.0)]
        if len(e1_full_dim) > 0:
            print(f"  [INFO] Found {len(e1_full_dim)} existing dim=16 runs from E1")
            print(f"  [INFO] Will reuse them and skip dim=16")

    results = []
    completed = 0
    errors = 0
    t_start = time.time()
    last_print_dim = None

    for dim in dimensions:
        # 尝试复用 E1 数据
        if dim == 16 and e1_df is not None and len(e1_full_dim) > 0:
            reuse_count = 0
            for _, row in e1_full_dim.iterrows():
                if row['target_id'] in [t.target_id for t in base_targets]:
                    results.append(ExperimentResult(
                        experiment='dimension',
                        scale=row['scale'], dim=dim, sparsity=sparsity,
                        algorithm=row['algorithm'], target_id=row['target_id'],
                        rep=int(row['rep']),
                        FHR=row['FHR'], BestFit=row['BestFit'], FSC=int(row['FSC']),
                        Diversity=row['Diversity'], Contrib=row['Contrib'],
                        CV=row['CV'], Entropy=row['Entropy'],
                        FFG=int(row['FFG']), n_evals=int(row['n_evals']),
                        n_foods=row['n_foods'], runtime=0.0
                    ))
                    reuse_count += 1
            completed += reuse_count
            print(f"\n  --- Dim {dim} [REUSED {reuse_count} from E1] ---")
            continue

        print(f"\n  --- Dim {dim} ---")

        # 投影食物空间
        projected_space = project_food_space(full_space, dim)

        for target in base_targets:
            # 投影 target
            projected_target = project_target(target, dim)

            for algorithm in algorithms:
                for rep in range(n_repeats):
                    task_seed = seed * 10000 + dim * 100 + rep

                    result = run_single(
                        projected_space, projected_target, algorithm,
                        scale, dim, sparsity, rep, task_seed
                    )
                    result.experiment = 'dimension'
                    results.append(result)
                    completed += 1

                    if result.FHR == 0.0 and result.BestFit == 0.0:
                        errors += 1

                    # 进度输出
                    if completed % 30 == 0 or completed == 1:
                        elapsed = time.time() - t_start
                        eta = elapsed / completed * (total_runs - completed) if completed > 0 else 0
                        pct = completed / total_runs * 100

                        if dim != last_print_dim:
                            last_print_dim = dim

                        print(f"  [{completed}/{total_runs}] {pct:.1f}% | "
                              f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s | "
                              f"errors={errors} | "
                              f"dim={dim} {algorithm} rep={rep} | "
                              f"FHR={result.FHR:.2f} Fit={result.BestFit:.3f} FSC={result.FSC}")

    total_time = time.time() - t_start

    # ============================================================
    # [4/4] 保存结果
    # ============================================================
    print(f"\n[4/4] Saving results...")
    print(f"  Completed: {completed}/{total_runs}")
    print(f"  Errors: {errors} ({errors / completed * 100:.1f}%)" if completed > 0 else "")
    print(f"  Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    df = pd.DataFrame([asdict(r) for r in results])
    output_path = os.path.join(output_dir, 'e3_dimension.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # ============================================================
    # 快速汇总
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Quick Summary: FHR by Dimension and Algorithm")
    print(f"{'=' * 70}")

    summary = df.groupby(['dim', 'algorithm'])['FHR'].mean().unstack()

    print(f"{'Dim':<6}", end="")
    for alg in algorithms:
        print(f"{alg:>10}", end="")
    print()
    print("-" * (6 + 10 * len(algorithms)))

    for dim in dimensions:
        print(f"{dim:<6}", end="")
        for alg in algorithms:
            if alg in summary.columns and dim in summary.index:
                val = summary.loc[dim, alg]
                print(f"{val:>10.3f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print("\nDone!")
    return df


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E3 Dimension Experiment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    run_e3_dimension(args.config)