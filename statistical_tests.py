#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计检验框架
用于100-200个target的实验后分析
包含: Friedman, Wilcoxon, Nemenyi, Cohen's d, 跨层Kruskal-Wallis
"""

import numpy as np
from scipy import stats

# ---------- 1. Friedman 检验（多算法排名） ----------
def friedman_test(results_matrix):
    """
    results_matrix: shape = (n_targets, n_algorithms)
    每行是一个target，每列是一个算法在该target上的性能值
    返回: statistic, p_value
    """
    from scipy.stats import friedmanchisquare

    # 将结果转换为排名（每行内排名，值越大排名越高）
    ranks = np.argsort(np.argsort(-results_matrix, axis=1), axis=1) + 1

    # Friedman检验
    stat, p = friedmanchisquare(*[results_matrix[:, i] for i in range(results_matrix.shape[1])])
    return stat, p

# ---------- 2. Wilcoxon 符号秩检验（两两对比） ----------
def wilcoxon_pairwise(results_matrix, algorithm_names, alpha=0.05, holm=True):
    """
    Pairwise Wilcoxon signed-rank tests with optional Holm family-wise correction.
    Returns: dict {(alg_i, alg_j): {statistic, p_value, p_holm, effect_size_r, significant}}
    """
    from scipy.stats import wilcoxon
    n_algs = results_matrix.shape[1]
    results = {}

    for i in range(n_algs):
        for j in range(i + 1, n_algs):
            diff = results_matrix[:, i] - results_matrix[:, j]
            nonzero = diff != 0
            if np.sum(nonzero) < 5:
                continue

            stat, p = wilcoxon(diff[nonzero])
            z = stats.norm.ppf(1 - p / 2) if p < 1 else 0
            n = np.sum(nonzero)
            r = z / np.sqrt(n) if n > 0 else 0

            results[(algorithm_names[i], algorithm_names[j])] = {
                "statistic": stat,
                "p_value": p,
                "p_holm": p,  # filled below
                "effect_size_r": r,
                "significant": p < alpha,
            }

    if holm and results:
        items = sorted(results.items(), key=lambda kv: kv[1]["p_value"])
        m = len(items)
        adj = []
        for rank, (key, val) in enumerate(items, start=1):
            p_adj = min(1.0, val["p_value"] * (m - rank + 1))
            adj.append((key, p_adj))
        # enforce monotonicity
        running = 0.0
        mono = []
        for key, p_adj in reversed(adj):
            running = max(running, p_adj)
            mono.append((key, running))
        for key, p_adj in reversed(mono):
            results[key]["p_holm"] = p_adj
            results[key]["significant"] = p_adj < alpha

    return results

# ---------- 3. Nemenyi 事后检验 ----------
def nemenyi_test(results_matrix, algorithm_names, alpha=0.05):
    """
    Nemenyi post-hoc test for Friedman
    判断哪些算法对之间有显著差异
    """
    n_targets, n_algs = results_matrix.shape

    # 转换为排名
    ranks = np.argsort(np.argsort(-results_matrix, axis=1), axis=1) + 1
    mean_ranks = np.mean(ranks, axis=0)

    # Nemenyi临界差值 (CD)
    q_alpha_table = {
        (2, 0.05): 1.960, (3, 0.05): 2.343, (4, 0.05): 2.569,
        (5, 0.05): 2.728, (6, 0.05): 2.850, (7, 0.05): 2.949,
        (8, 0.05): 3.031, (9, 0.05): 3.102, (10, 0.05): 3.164,
    }
    q_alpha = q_alpha_table.get((n_algs, alpha), 3.0)
    cd = q_alpha * np.sqrt(n_algs * (n_algs + 1) / (6 * n_targets))

    # 判断差异
    significant_pairs = []
    for i in range(n_algs):
        for j in range(i+1, n_algs):
            if abs(mean_ranks[i] - mean_ranks[j]) > cd:
                significant_pairs.append((
                    algorithm_names[i], algorithm_names[j],
                    abs(mean_ranks[i] - mean_ranks[j]), cd
                ))

    return {
        'mean_ranks': dict(zip(algorithm_names, mean_ranks)),
        'critical_difference': cd,
        'significant_pairs': significant_pairs
    }

# ---------- 4. Cohen's d 效应量 ----------
def cohens_d(x, y):
    """
    计算Cohen's d效应量
    d < 0.2: negligible
    0.2 <= d < 0.5: small
    0.5 <= d < 0.8: medium
    d >= 0.8: large
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2

    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)

    d = (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
    return d

def interpret_cohens_d(d):
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'

# ---------- 5. 跨层分析 (按营养素维度分层) ----------
def cross_layer_analysis(results_by_layer, algorithm_names):
    """
    分析算法在不同营养素维度层的表现差异
    results_by_layer: {'low': results_matrix_low, 'medium': ..., 'high': ...}
    """
    from scipy.stats import kruskal

    print("\n【跨层性能分析】")
    for layer, matrix in results_by_layer.items():
        mean_perf = np.mean(matrix, axis=0)
        std_perf = np.std(matrix, axis=0)
        print(f"\n{layer}层 (n={matrix.shape[0]} targets):")
        for i, (m, s) in enumerate(zip(mean_perf, std_perf)):
            print(f"  {algorithm_names[i]}: mean={m:.4f} ± {s:.4f}")

    # Kruskal-Wallis检验（跨层比较同一算法）
    n_algs = list(results_by_layer.values())[0].shape[1]

    print(f"\n【Kruskal-Wallis 跨层检验】")
    for i in range(n_algs):
        samples = [matrix[:, i] for matrix in results_by_layer.values()]
        stat, p = kruskal(*samples)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {algorithm_names[i]}: H={stat:.3f}, p={p:.4f} {sig}")


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    # 假设有3个算法在100个target上的结果
    np.random.seed(42)
    n_targets = 100
    n_algorithms = 3
    algorithm_names = ['Alg_A', 'Alg_B', 'Alg_C']

    # 模拟结果 (越大越好)
    results = np.random.rand(n_targets, n_algorithms)
    results[:, 1] += 0.05  # Alg_B 略好

    # 1. Friedman检验
    stat, p = friedman_test(results)
    print(f"Friedman: chi2={stat:.3f}, p={p:.4f}")

    # 2. 两两Wilcoxon
    pairwise = wilcoxon_pairwise(results, algorithm_names)
    for (a1, a2), res in pairwise.items():
        print(f"{a1} vs {a2}: p={res['p_value']:.4f}, r={res['effect_size_r']:.3f}")

    # 3. Nemenyi
    nemenyi = nemenyi_test(results, algorithm_names)
    print(f"Mean ranks: {nemenyi['mean_ranks']}")
    print(f"CD = {nemenyi['critical_difference']:.3f}")

    # 4. Cohen's d (Alg_A vs Alg_B)
    d = cohens_d(results[:, 0], results[:, 1])
    print(f"Cohen's d = {d:.3f} ({interpret_cohens_d(d)})")
