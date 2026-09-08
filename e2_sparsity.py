#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2 — Sparsity Experiment
目的: 测试约束松紧度对可行域和算法行为的影响
设计: 3 sparsity × 7 algorithms × 12 targets × 10 repeats = 2,520 runs

重用 E1 的所有组件：FoodSpace, Target, ExperimentResult, ALGO_RUNNERS, run_single
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from dataclasses import asdict
import warnings

warnings.filterwarnings('ignore')

# 从 E1 导入所有可重用组件
from e1_scaling import (
    Target, ExperimentResult,
    load_nhanes_full, load_targets,
    run_single
)


# 从共享模块导入（含无上界清洗）
from sparsity_ops import adjust_target_object, empirical_meal_caps


# ============================================================
# E2 特有：约束调整（兼容旧接口；生产请用 run_e2.py）
# ============================================================

def adjust_target_sparsity(target: Target, factor: float, caps: np.ndarray = None) -> Target:
    """
    调整 target 的约束松紧度（先清洗 sentinel 上界，再按中心缩放）。
    factor=1.0: 清洗后的基准区间
    factor=0.5: 收紧
    factor=1.5: 放松
    """
    if caps is None:
        # Fallback: treat huge uppers as lo + max(lo,1)*10 — weak; prefer passing caps
        hi = target.upper_bounds.astype(float).copy()
        lo = target.lower_bounds.astype(float).copy()
        caps = np.where(hi >= 1e5, np.maximum(lo * 2 + 1, lo + 100.0), hi)
    adj, _ = adjust_target_object(target, factor, caps)
    return adj


# ============================================================
# E2 主实验
# ============================================================

def run_e2_sparsity(config_path: str = 'config.yaml'):
    """E2 主实验"""

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    algorithms = config['algorithms']
    sparsity_factors = config.get('sparsity_factors', [0.5, 1.0, 1.5])
    n_targets = config.get('targets_total', 12)
    n_repeats = config.get('repeats', 10)
    seed = config.get('seed', 42)
    output_dir = config.get('output_dir', 'results/raw')

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # [1/4] 加载数据
    # ============================================================
    print("=" * 70)
    print("E2 — Sparsity Experiment")
    print("=" * 70)
    print(f"\n[1/4] Loading NHANES full data...")

    full_space = load_nhanes_full()
    print(f"  Full space: {full_space.n_foods} foods, {full_space.n_nutrients} nutrients")

    # E2 固定使用 210K 全量
    food_space = full_space
    scale = full_space.n_foods
    dim = full_space.n_nutrients

    # ============================================================
    # [2/4] 加载 targets
    # ============================================================
    print(f"\n[2/4] Loading targets...")
    base_targets = load_targets(n_targets, seed=seed)
    print(f"  Selected {len(base_targets)} targets")
    for t in base_targets[:3]:
        print(f"    {t.target_id}: {t.n_dims} dims")
    if len(base_targets) > 3:
        print(f"    ... and {len(base_targets) - 3} more")

    # ============================================================
    # [3/4] 运行实验
    # ============================================================
    print(f"\n[3/4] Running experiments...")
    print(f"  Sparsity factors: {sparsity_factors}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Targets: {len(base_targets)}")
    print(f"  Repeats: {n_repeats}")
    total_runs = len(sparsity_factors) * len(algorithms) * len(base_targets) * n_repeats
    print(f"  Total runs: {total_runs}")

    # 检查是否可以复用 E1 的 normal 数据
    e1_path = os.path.join(output_dir, 'e1_scaling.csv')
    e1_df = None
    if os.path.exists(e1_path):
        e1_df = pd.read_csv(e1_path)
        e1_normal = e1_df[(e1_df['scale'] == 210000) & (e1_df['sparsity'] == 1.0)]
        if len(e1_normal) > 0:
            print(f"  [INFO] Found {len(e1_normal)} existing normal-sparsity runs from E1")
            print(f"  [INFO] Will reuse them and skip sparsity=1.0")

    results = []
    completed = 0
    errors = 0
    t_start = time.time()
    last_print_sparsity = None

    for sparsity in sparsity_factors:
        sparsity_label = {0.5: 'tight', 1.0: 'normal', 1.5: 'loose'}.get(sparsity, str(sparsity))

        # 尝试复用 E1 数据
        if sparsity == 1.0 and e1_df is not None and len(e1_normal) > 0:
            reuse_count = 0
            for _, row in e1_normal.iterrows():
                if row['target_id'] in [t.target_id for t in base_targets]:
                    results.append(ExperimentResult(
                        experiment='sparsity',
                        scale=row['scale'], dim=row['dim'], sparsity=sparsity,
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
            print(f"\n  --- Sparsity {sparsity} ({sparsity_label}) [REUSED {reuse_count} from E1] ---")
            continue

        print(f"\n  --- Sparsity {sparsity} ({sparsity_label}) ---")

        for target in base_targets:
            # 调整约束
            adjusted_target = adjust_target_sparsity(target, sparsity)

            for algorithm in algorithms:
                for rep in range(n_repeats):
                    task_seed = seed * 10000 + int(sparsity * 100) * 100 + rep

                    result = run_single(
                        food_space, adjusted_target, algorithm,
                        scale, dim, sparsity, rep, task_seed
                    )
                    result.experiment = 'sparsity'
                    results.append(result)
                    completed += 1

                    if result.FHR == 0.0 and result.BestFit == 0.0:
                        errors += 1

                    # 进度输出（每 30 次或换 sparsity 时）
                    if completed % 30 == 0 or completed == 1:
                        elapsed = time.time() - t_start
                        eta = elapsed / completed * (total_runs - completed) if completed > 0 else 0
                        pct = completed / total_runs * 100

                        if sparsity != last_print_sparsity:
                            last_print_sparsity = sparsity

                        print(f"  [{completed}/{total_runs}] {pct:.1f}% | "
                              f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s | "
                              f"errors={errors} | "
                              f"sparsity={sparsity} {algorithm} rep={rep} | "
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
    output_path = os.path.join(output_dir, 'e2_sparsity.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # ============================================================
    # 快速汇总
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Quick Summary: FHR by Sparsity and Algorithm")
    print(f"{'=' * 70}")

    summary = df.groupby(['sparsity', 'algorithm'])['FHR'].mean().unstack()

    sparsity_names = {0.5: 'tight', 1.0: 'normal', 1.5: 'loose'}
    print(f"{'Sparsity':<12}", end="")
    for alg in algorithms:
        print(f"{alg:>10}", end="")
    print()
    print("-" * (12 + 10 * len(algorithms)))

    for sparsity in sparsity_factors:
        label = sparsity_names.get(sparsity, f"{sparsity:.1f}")
        print(f"{label:<12}", end="")
        for alg in algorithms:
            if alg in summary.columns and sparsity in summary.index:
                val = summary.loc[sparsity, alg]
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

    parser = argparse.ArgumentParser(description='E2 Sparsity Experiment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    run_e2_sparsity(args.config)