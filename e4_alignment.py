#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E4 — Cross-Study Alignment Analysis
目的: 统一分析 E1-E3 结果，验证跨实验规律一致性
输入: E1 scaling.csv + E2 sparsity.csv + E3 dimension.csv
输出: 统一趋势曲线、ADI、Regime Boundary Consistency
类型: 纯分析层（不产生新实验数据）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
from scipy.optimize import curve_fit
from scipy.stats import kendalltau
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 8),
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

INPUT_DIR = 'results/raw'
OUTPUT_DIR = 'results/e4_analysis'
FIG_DIR = 'results/figures'


# ============================================================
# 1. 数据加载
# ============================================================

def load_all_experiments():
    """加载 E1-E3 所有数据"""
    dfs = {}

    for exp_name, filename in [
        ('scaling', 'e1_scaling_full.csv'),
        ('sparsity', 'e2_sparsity_full.csv'),
        ('dimension', 'e3_dimension_full.csv'),
    ]:
        path = os.path.join(INPUT_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # E1: analyze NHANES scaling axis separately from recipe catalogs
            if exp_name == 'scaling' and 'dataset' in df.columns:
                n0 = len(df)
                df = df[df['dataset'].astype(str).str.startswith('nhanes')].copy()
                print(f"  Loaded {exp_name}: {len(df)} NHANES rows (dropped {n0 - len(df)} recipe rows)")
            else:
                print(f"  Loaded {exp_name}: {len(df)} rows")
            df['experiment'] = exp_name
            dfs[exp_name] = df
        else:
            print(f"  [WARNING] {path} not found, skipping {exp_name}")

    return dfs


def build_unified_frame(dfs):
    """合并所有实验数据并归一化"""
    unified = pd.concat(dfs.values(), ignore_index=True)

    # 归一化：按 experiment 分组，min-max 归一化
    metrics = ['FHR', 'BestFit', 'FSC', 'Diversity']
    for metric in metrics:
        col_name = f'{metric}_norm'
        unified[col_name] = unified.groupby('experiment')[metric].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
        )

    # 对数规模轴（用于 scaling 实验）
    unified['log_scale'] = np.log10(unified['scale'])

    return unified


# ============================================================
# 2. 分析1: Universal Curve Collapse
# ============================================================

def compute_curve_collapse(unified):
    """检查不同实验条件下曲线是否收敛到统一形态"""
    print("\n" + "=" * 70)
    print("Analysis 1: Universal Curve Collapse")
    print("=" * 70)

    # 只对 scaling 实验做（因为 sparsity/dimension 的 x 轴含义不同）
    scaling_df = unified[unified['experiment'] == 'scaling']

    if len(scaling_df) == 0:
        print("  [SKIP] No scaling data")
        return None, None

    # 按 log_scale 和 algorithm 聚合
    grouped = scaling_df.groupby(['algorithm', 'log_scale'])['FHR_norm'].agg(['mean', 'std']).reset_index()

    # 每个 x 点上的跨算法方差（衡量不同算法是否呈现一致趋势）
    sigma_by_x = grouped.groupby('log_scale')['mean'].var()
    sigma_align = sigma_by_x.mean()

    print(f"  Cross-algorithm variance (sigma_align): {sigma_align:.4f}")
    print(
        f"  Interpretation: {'Strong universality' if sigma_align < 0.01 else 'Moderate consistency' if sigma_align < 0.05 else 'Algorithm-dependent behavior'}")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    algorithms = scaling_df['algorithm'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

    for i, alg in enumerate(algorithms):
        alg_data = grouped[grouped['algorithm'] == alg]
        ax.plot(alg_data['log_scale'], alg_data['mean'],
                color=colors[i], label=alg, linewidth=2, marker='o', markersize=4)
        ax.fill_between(alg_data['log_scale'],
                        alg_data['mean'] - alg_data['std'],
                        alg_data['mean'] + alg_data['std'],
                        color=colors[i], alpha=0.15)

    ax.set_xlabel('log10(Food Space Size)', fontsize=13)
    ax.set_ylabel('Normalized FHR', fontsize=13)
    ax.set_title('E4-1: Universal Curve Collapse — Scaling Behavior', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_e4_1_curve_collapse.png'))
    plt.close(fig)
    print(f"  Saved: fig_e4_1_curve_collapse.png")

    return grouped, sigma_align


# ============================================================
# 3. 分析2: Alignment Deviation Index (ADI)
# ============================================================

def compute_adi(unified):
    """计算跨实验 Alignment Deviation Index"""
    print("\n" + "=" * 70)
    print("Analysis 2: Alignment Deviation Index (ADI)")
    print("=" * 70)

    algorithms = unified['algorithm'].unique()
    experiments = unified['experiment'].unique()

    # 全局均值
    global_mean = unified.groupby(['experiment', 'algorithm'])['FHR_norm'].mean().reset_index()

    # 对每个实验-算法组合，计算与全局均值的偏差
    adi_results = []
    for exp in experiments:
        exp_data = unified[unified['experiment'] == exp]
        exp_mean = exp_data.groupby('algorithm')['FHR_norm'].mean()

        for alg in algorithms:
            if alg in exp_mean.index:
                # 计算该算法在其他实验中的均值
                other_data = unified[(unified['experiment'] != exp) & (unified['algorithm'] == alg)]
                if len(other_data) > 0:
                    other_mean = other_data['FHR_norm'].mean()
                    deviation = abs(exp_mean[alg] - other_mean)
                    adi_results.append({
                        'experiment': exp,
                        'algorithm': alg,
                        'ADI': deviation
                    })

    adi_df = pd.DataFrame(adi_results)

    # 按算法汇总
    adi_summary = adi_df.groupby('algorithm')['ADI'].mean().sort_values()

    print(f"\n  ADI by Algorithm (lower = more consistent across experiments):")
    for alg, adi in adi_summary.items():
        level = '★★★ Strong universality' if adi < 0.05 else '★★  Moderate' if adi < 0.15 else '★   Dataset-dependent'
        print(f"    {alg:<12}: ADI={adi:.4f}  {level}")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(adi_summary)), adi_summary.values,
                  color=plt.cm.RdYlGn_r(adi_summary.values / max(adi_summary.values, default=1)))
    ax.set_xticks(range(len(adi_summary)))
    ax.set_xticklabels(adi_summary.index, rotation=45, ha='right')
    ax.set_ylabel('Alignment Deviation Index', fontsize=13)
    ax.set_title('E4-2: Cross-Experiment Consistency (ADI)', fontsize=14)
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Strong (< 0.05)')
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate (< 0.15)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_e4_2_adi_bar.png'))
    plt.close(fig)
    print(f"  Saved: fig_e4_2_adi_bar.png")

    return adi_summary


# ============================================================
# 4. 分析3: Regime Boundary Consistency
# ============================================================

def logistic(x, L, k, x0):
    """Logistic 函数用于相变拟合"""
    return L / (1 + np.exp(k * (x - x0)))


def fit_phase_transition(df, alg_column='algorithm', value_column='FHR'):
    """对每个算法拟合相变曲线"""
    algorithms = df[alg_column].unique()
    params = {}

    for alg in algorithms:
        alg_data = df[df[alg_column] == alg]
        if len(alg_data) < 5:
            continue

        # 按 scale 聚合
        summary = alg_data.groupby('scale')[value_column].mean().reset_index()
        x = np.log10(summary['scale'].values)
        y = summary[value_column].values

        try:
            popt, _ = curve_fit(logistic, x, y, p0=[1.0, 1.0, 4.0], maxfev=5000)
            L, k, x0 = popt
            N_star = 10 ** x0
            y_pred = logistic(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            params[alg] = {'N_star': N_star, 'k': k, 'R2': r2}
        except:
            params[alg] = {'N_star': np.nan, 'k': np.nan, 'R2': 0.0}

    return pd.DataFrame(params).T


def compare_transition_boundaries(unified):
    """比较不同实验的相变临界点"""
    print("\n" + "=" * 70)
    print("Analysis 3: Regime Boundary Consistency")
    print("=" * 70)

    # 只对 scaling 实验做相变分析
    scaling_df = unified[unified['experiment'] == 'scaling']

    if len(scaling_df) == 0:
        print("  [SKIP] No scaling data")
        return None

    params_df = fit_phase_transition(scaling_df)

    print(f"\n  Phase Transition Parameters (from Scaling Experiment):")
    print(f"  {'Algorithm':<12} {'N*':>10} {'k (steepness)':>14} {'R²':>8}")
    print(f"  {'-' * 44}")
    for alg, row in params_df.iterrows():
        print(f"  {alg:<12} {row['N_star']:>10.0f} {row['k']:>14.2f} {row['R2']:>8.3f}")

    # 绘图
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    algorithms = scaling_df['algorithm'].unique()

    for i, alg in enumerate(algorithms[:8]):
        ax = axes[i]
        alg_data = scaling_df[scaling_df['algorithm'] == alg]
        summary = alg_data.groupby('scale')['FHR'].agg(['mean', 'std']).reset_index()

        x = np.log10(summary['scale'].values)
        y = summary['mean'].values

        # 数据点
        ax.errorbar(x, y, yerr=summary['std'].values, fmt='o', color='steelblue',
                    capsize=3, markersize=5, label='Observed')

        # 拟合曲线
        if alg in params_df.index and not np.isnan(params_df.loc[alg, 'N_star']):
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = logistic(x_fit, 1.0, params_df.loc[alg, 'k'], np.log10(params_df.loc[alg, 'N_star']))
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f"N*={params_df.loc[alg, 'N_star']:.0f}")
            ax.axvline(x=np.log10(params_df.loc[alg, 'N_star']), color='red',
                       linestyle='--', alpha=0.5)

        ax.set_title(alg, fontsize=12)
        ax.set_xlabel('log10(Scale)')
        ax.set_ylabel('FHR')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(len(algorithms), 8):
        axes[j].set_visible(False)

    fig.suptitle('E4-3: Phase Transition Fitting per Algorithm', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_e4_3_phase_transition.png'))
    plt.close(fig)
    print(f"  Saved: fig_e4_3_phase_transition.png")

    return params_df


# ============================================================
# 5. Ranking Stability (Kendall's τ)
# ============================================================

def compute_ranking_stability(unified):
    """计算 Small-scale vs Large-scale 的排名一致性"""
    print("\n" + "=" * 70)
    print("Analysis: Ranking Stability (Kendall's τ)")
    print("=" * 70)

    scaling_df = unified[unified['experiment'] == 'scaling']
    if len(scaling_df) == 0:
        print("  [SKIP] No scaling data")
        return None

    # 提取最小规模和最大规模的排名
    min_scale = scaling_df['scale'].min()
    max_scale = scaling_df['scale'].max()

    small_rank = scaling_df[scaling_df['scale'] == min_scale].groupby('algorithm')['FHR'].mean()
    large_rank = scaling_df[scaling_df['scale'] == max_scale].groupby('algorithm')['FHR'].mean()

    # 计算 Kendall's τ
    common_algs = small_rank.index.intersection(large_rank.index)
    if len(common_algs) >= 3:
        tau, p = kendalltau(
            small_rank[common_algs].rank(ascending=False),
            large_rank[common_algs].rank(ascending=False)
        )
        print(f"  Kendall's τ = {tau:.3f} (p={p:.4f})")
        if p < 0.05:
            print(f"  ★ Rankings are SIGNIFICANTLY CONSISTENT across scales")
        else:
            print(f"  ★ Rankings are NOT significantly consistent — REVERSAL detected!")
    else:
        print(f"  [SKIP] Not enough common algorithms ({len(common_algs)})")
        tau, p = np.nan, np.nan

    return {'tau': tau, 'p': p, 'small_rank': small_rank, 'large_rank': large_rank}


# ============================================================
# 6. 主分析流程
# ============================================================

def run_e4_analysis():
    """E4 主分析"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("E4 — Cross-Study Alignment Analysis")
    print("=" * 70)

    # 加载数据
    print(f"\n[1/5] Loading experiment data...")
    dfs = load_all_experiments()

    if len(dfs) == 0:
        print("  [ERROR] No data found. Run E1-E3 first.")
        return

    # 构建统一框架
    print(f"\n[2/5] Building unified frame...")
    unified = build_unified_frame(dfs)
    print(f"  Total rows: {len(unified)}")

    # 分析1: Curve Collapse
    print(f"\n[3/5] Analysis 1: Curve Collapse...")
    collapse_data, sigma_align = compute_curve_collapse(unified)

    # 分析2: ADI
    print(f"\n[4/5] Analysis 2: ADI...")
    adi_summary = compute_adi(unified)

    # 分析3: Regime Boundary
    print(f"\n[5/5] Analysis 3: Regime Boundary...")
    params_df = compare_transition_boundaries(unified)

    # 排名稳定性
    ranking = compute_ranking_stability(unified)

    # 保存汇总
    summary = {
        'sigma_align': sigma_align if sigma_align else np.nan,
        'kendall_tau': ranking['tau'] if ranking else np.nan,
        'kendall_p': ranking['p'] if ranking else np.nan,
    }

    if adi_summary is not None:
        adi_summary.to_csv(os.path.join(OUTPUT_DIR, 'adi_results.csv'))

    if params_df is not None:
        params_df.to_csv(os.path.join(OUTPUT_DIR, 'phase_transition_params.csv'))

    pd.Series(summary).to_csv(os.path.join(OUTPUT_DIR, 'e4_summary.csv'))

    print(f"\n{'=' * 70}")
    print("E4 Analysis Complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"  Figures saved to: {FIG_DIR}")
    print(f"{'=' * 70}")

    return unified


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    run_e4_analysis()