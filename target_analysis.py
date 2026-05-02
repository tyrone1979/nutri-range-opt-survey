#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target 分析与代表性抽取系统
功能：
1. 全量 target 的营养成分维度分析
2. 按营养素维度数分层（低2-4种 / 中5-8种 / 高9+种）
3. 每层内部聚类抽取代表性 target
4. 确保每层统计显著性（n>=30）
5. 输出统计检验建议与代码框架

作者: AI Assistant
"""

import pandas as pd
import numpy as np
import json
import ast
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# ============================================================
# 1. 数据加载
# ============================================================

def load_targets(csv_path: str) -> List[Dict]:
    """从CSV加载所有target字典"""
    df = pd.read_csv(csv_path)
    targets = []
    for _, row in df.iterrows():
        try:
            t = ast.literal_eval(row['target'])
            targets.append(t)
        except Exception as e:
            print(f"解析失败: {e}")
    return targets

# ============================================================
# 2. 维度统计与分层
# ============================================================

# 16种营养成分字段（按常见顺序）
NUTRIENT_FIELDS = [
    'calorie', 'protein', 'carb', 'sugar', 'saturated_fat', 'cholesterol',
    'sodium', 'potassium', 'fiber', 'calcium', 'vitamin_d', 'vitamin_c',
    'iron', 'folic_acid', 'vitamin_b12', 'phosphorus'
]

def analyze_dimensions(targets: List[Dict]) -> Dict:
    """分析每个target的营养素维度数，并分层"""
    dim_stats = {'low': [], 'medium': [], 'high': []}

    for idx, t in enumerate(targets):
        n_dims = len(t.keys())
        if 2 <= n_dims <= 4:
            dim_stats['low'].append(idx)
        elif 5 <= n_dims <= 8:
            dim_stats['medium'].append(idx)
        elif n_dims >= 9:
            dim_stats['high'].append(idx)

    return dim_stats

def print_dimension_report(targets: List[Dict], dim_stats: Dict):
    """打印维度分层报告"""
    total = len(targets)
    print("=" * 70)
    print("【营养素维度分层统计】")
    print("=" * 70)

    for layer_name, layer_label, idx_list in [
        ('low', '低维度 (2-4种)', dim_stats['low']),
        ('medium', '中维度 (5-8种)', dim_stats['medium']),
        ('high', '高维度 (9+种)', dim_stats['high']),
    ]:
        n = len(idx_list)
        print(f"\n{layer_label}: {n} 个 ({n/total*100:.1f}%)")

        # 统计该层的字段分布
        field_counts = Counter()
        for idx in idx_list:
            field_counts.update(targets[idx].keys())

        print("  字段分布:")
        for field, cnt in field_counts.most_common():
            print(f"    {field:<15}: {cnt:4d} ({cnt/n*100:5.1f}%)")

# ============================================================
# 3. 特征工程
# ============================================================

def build_feature_matrix(targets: List[Dict], field_list: List[str]) -> np.ndarray:
    """
    将target列表转为特征矩阵
    每维提取 [lower, upper]，缺失用全局中位数填充
    9999999.0 无限上限用合理值替代
    """
    # 计算全局统计
    stats = {}
    for f in field_list:
        lowers, uppers = [], []
        for t in targets:
            if f in t:
                lowers.append(t[f][0])
                uppers.append(t[f][1])
        stats[f] = {
            'lower_median': np.median(lowers) if lowers else 0,
            'upper_median': np.median(uppers) if uppers else 0,
            'upper_reasonable_max': max([u for u in uppers if u < 999999]) if uppers else 0
        }

    # 构建特征矩阵
    X = []
    for t in targets:
        features = []
        for f in field_list:
            if f in t:
                lo, up = t[f][0], t[f][1]
                if up >= 999999:
                    up = stats[f]['upper_reasonable_max'] * 1.2
            else:
                lo = stats[f]['lower_median']
                up = stats[f]['upper_median']
            features.append(lo)
            features.append(up)
        X.append(features)

    return np.array(X), stats

# ============================================================
# 4. 分层聚类与代表性抽取
# ============================================================

@dataclass
class Representative:
    rep_id: int
    layer: str          # 'low'/'medium'/'high'
    cluster_id: int
    original_idx: int   # 在全量targets中的索引
    size: int           # 该簇样本数
    strength: str       # 'strong'(>=30) / 'moderate'(25-29) / 'weak'(20-24)
    features: List[str]
    target: Dict

def cluster_and_extract(
    targets: List[Dict],
    indices: List[int],
    layer_name: str,
    field_list: List[str],
    n_clusters: int,
    min_cluster_size: int = 20
) -> List[Representative]:
    """
    对指定层进行聚类并抽取代表性target
    """
    layer_targets = [targets[i] for i in indices]
    n_samples = len(layer_targets)

    if n_samples < n_clusters:
        n_clusters = max(2, n_samples // min_cluster_size)

    # 构建特征矩阵
    X_local, stats_local = build_feature_matrix(layer_targets, field_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_local)

    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_

    # 统计簇大小
    counts = Counter(labels)
    print(f"\n【{layer_name}】初始聚类: {n_clusters} 簇")
    for i in range(n_clusters):
        print(f"  簇 {i}: {counts[i]} 样本")

    # 小簇合并（< min_cluster_size）
    new_labels = labels.copy()
    small_clusters = [i for i in range(n_clusters) if counts[i] < min_cluster_size]
    large_clusters = [i for i in range(n_clusters) if counts[i] >= min_cluster_size]

    for sc in small_clusters:
        sc_indices = np.where(labels == sc)[0]
        sc_points = X_scaled[sc_indices]
        if len(large_clusters) == 0:
            break
        large_centers = centers[large_clusters]
        dists = cdist(sc_points, large_centers, metric='euclidean')
        avg_dists = np.mean(dists, axis=0)
        nearest = large_clusters[np.argmin(avg_dists)]
        new_labels[sc_indices] = nearest

    # 重新统计
    final_counts = Counter(new_labels)
    final_ids = sorted(set(new_labels))

    # 为每个最终簇找最近样本
    reps = []
    rep_counter = 0

    for cid in final_ids:
        cluster_indices_local = np.where(new_labels == cid)[0]
        cluster_indices_global = [indices[i] for i in cluster_indices_local]
        cluster_points = X_scaled[cluster_indices_local]

        # 重新计算质心
        center = np.mean(cluster_points, axis=0)

        # 找最近样本
        dists = cdist(cluster_points, center.reshape(1, -1), metric='euclidean').flatten()
        closest_local = cluster_indices_local[np.argmin(dists)]
        closest_global = indices[closest_local]

        size = len(cluster_indices_global)
        if size >= 30:
            strength = 'strong'
        elif size >= 25:
            strength = 'moderate'
        elif size >= 20:
            strength = 'weak'
        else:
            strength = 'insufficient'

        # 生成特征标签
        t = targets[closest_global]
        features = generate_feature_tags(t)

        reps.append(Representative(
            rep_id=rep_counter,
            layer=layer_name,
            cluster_id=cid,
            original_idx=closest_global,
            size=size,
            strength=strength,
            features=features,
            target=t
        ))
        rep_counter += 1

    return reps

def generate_feature_tags(t: Dict) -> List[str]:
    """为target生成可读的特征标签"""
    tags = []

    # 卡路里
    if 'calorie' in t:
        cal = t['calorie']
        if cal[1] < 2500 and cal[0] > 1000:
            tags.append("低卡固定")
        elif cal[1] < 2500:
            tags.append("低卡")
        elif cal[1] < 4000:
            tags.append("中卡")
        else:
            tags.append("高卡")
        if cal[0] > 0:
            tags.append(f"卡下限{cal[0]}")

    # 蛋白质
    if 'protein' in t:
        pro = t['protein']
        if pro[1] > 100:
            tags.append("高蛋白")
        elif pro[1] < 70:
            tags.append("低蛋白")
        if pro[0] > 50:
            tags.append(f"高下限蛋白{pro[0]}+")

    # 碳水
    if 'carb' in t:
        carb = t['carb']
        if carb[1] > 600:
            tags.append("高碳水")
        if carb[0] > 300:
            tags.append(f"高下限碳水{carb[0]}+")
        elif carb[0] == 0:
            tags.append("碳水无下限")

    # 糖
    if 'sugar' in t:
        if t['sugar'][1] <= 25:
            tags.append("严格控糖<=25")
        else:
            tags.append("控糖<=36")
    else:
        tags.append("无糖限制")

    # 饱和脂肪
    if 'saturated_fat' in t:
        sf = t['saturated_fat']
        if sf[1] >= 99999:
            tags.append("SatFat无上限")
        elif sf[1] <= 20:
            tags.append("低SatFat<=20")
        else:
            tags.append(f"SatFat<={sf[1]:.0f}")

    # 胆固醇
    if 'cholesterol' in t and t['cholesterol'][1] <= 200:
        tags.append("低胆固醇<=200")

    # 纤维
    if 'fiber' in t:
        tags.append(f"纤维>={t['fiber'][0]}")

    # 钠/钾
    if 'sodium' in t:
        tags.append("限钠")
    if 'potassium' in t:
        tags.append("高钾")

    # 微量营养素
    if 'calcium' in t and 'vitamin_d' in t:
        tags.append("钙+维D")
    if 'iron' in t and 'folic_acid' in t and 'vitamin_b12' in t:
        tags.append("孕期(铁+叶酸+B12)")
    elif 'iron' in t or 'folic_acid' in t:
        tags.append("铁/叶酸补充")
    if 'vitamin_c' in t:
        tags.append("维C")

    return tags

# ============================================================
# 5. 分层配额计算
# ============================================================

def calculate_quotas(n_total: int, dim_stats: Dict, min_per_layer: int = 10) -> Dict:
    """
    按比例分配每层抽取数，但确保每层至少 min_per_layer 个
    """
    total_samples = sum(len(v) for v in dim_stats.values())

    # 先按样本比例分配
    raw_quotas = {}
    for layer in ['low', 'medium', 'high']:
        raw_quotas[layer] = int(n_total * len(dim_stats[layer]) / total_samples)

    # 保底 + 余量再分配
    quotas = {}
    remaining = n_total

    for layer in ['low', 'medium', 'high']:
        quotas[layer] = max(min_per_layer, raw_quotas[layer])
        remaining -= quotas[layer]

    # 余量按样本量比例分配
    if remaining > 0:
        for layer in ['high', 'medium', 'low']:
            extra = min(remaining, len(dim_stats[layer]) // 30)
            quotas[layer] += extra
            remaining -= extra
            if remaining <= 0:
                break

    return quotas

# ============================================================
# 6. 主流程
# ============================================================

def main(csv_path: str, n_representatives: int = 100, output_path: str = None):
    """主流程"""

    # 1. 加载
    print("=" * 70)
    print("【Step 1】加载数据")
    targets = load_targets(csv_path)
    print(f"总样本: {len(targets)}")

    # 2. 维度分析
    print("\n" + "=" * 70)
    print("【Step 2】维度分层分析")
    dim_stats = analyze_dimensions(targets)
    print_dimension_report(targets, dim_stats)

    # 3. 计算配额
    print("\n" + "=" * 70)
    print("【Step 3】计算分层配额")
    quotas = calculate_quotas(n_representatives, dim_stats, min_per_layer=10)
    for layer, q in quotas.items():
        print(f"  {layer}层: 配额 {q} 个代表")

    # 4. 分层聚类抽取
    print("\n" + "=" * 70)
    print("【Step 4】分层聚类抽取")

    all_reps = []
    for layer, quota in quotas.items():
        if quota <= 0 or len(dim_stats[layer]) == 0:
            continue

        n_samples = len(dim_stats[layer])
        n_clusters = min(quota * 2, max(2, n_samples // 25))

        print(f"\n>>> {layer}层: {n_samples}样本 -> 聚类{n_clusters} -> 抽取{quota}个")

        reps = cluster_and_extract(
            targets=targets,
            indices=dim_stats[layer],
            layer_name=layer,
            field_list=NUTRIENT_FIELDS,
            n_clusters=n_clusters,
            min_cluster_size=20
        )

        reps_sorted = sorted(reps, key=lambda x: x.size, reverse=True)
        selected = reps_sorted[:quota]

        print(f"  生成 {len(reps)} 个簇，选取前 {len(selected)} 个")

        for i, r in enumerate(selected):
            r.rep_id = len(all_reps) + i

        all_reps.extend(selected)

    # 5. 统计报告
    print("\n" + "=" * 70)
    print("【Step 5】统计验证")

    total_covered = sum(r.size for r in all_reps)
    strong = len([r for r in all_reps if r.strength == 'strong'])
    moderate = len([r for r in all_reps if r.strength == 'moderate'])
    weak = len([r for r in all_reps if r.strength == 'weak'])

    print(f"  总代表数: {len(all_reps)}")
    print(f"  强统计(>=30): {strong} ({strong/len(all_reps)*100:.1f}%)")
    print(f"  中统计(25-29): {moderate} ({moderate/len(all_reps)*100:.1f}%)")
    print(f"  弱统计(20-24): {weak} ({weak/len(all_reps)*100:.1f}%)")
    print(f"  覆盖样本: {total_covered}/{len(targets)} ({total_covered/len(targets)*100:.1f}%)")

    for layer in ['low', 'medium', 'high']:
        layer_reps = [r for r in all_reps if r.layer == layer]
        if layer_reps:
            print(f"\n  {layer}层代表:")
            for r in layer_reps:
                print(f"    [{r.rep_id:3d}] {r.strength:8s} size={r.size:3d} | {' | '.join(r.features[:5])}")

    # 6. 保存
    if output_path:
        print("\n" + "=" * 70)
        print("【Step 6】保存结果")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rep_id', 'layer', 'cluster_id', 'original_idx', 'cluster_size',
                'statistical_strength', 'tags', 'target_json'
            ])

            for r in all_reps:
                writer.writerow([
                    r.rep_id,
                    r.layer,
                    r.cluster_id,
                    r.original_idx,
                    r.size,
                    r.strength,
                    ' | '.join(r.features),
                    json.dumps(r.target, ensure_ascii=False, separators=(',', ':'))
                ])

        print(f"已保存到: {output_path}")

    return all_reps

# ============================================================
# 7. 统计检验框架（独立文件内容）
# ============================================================

STATISTICAL_TESTS_CODE = r"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
统计检验框架
用于100-200个target的实验后分析
包含: Friedman, Wilcoxon, Nemenyi, Cohen's d
\"\"\"

import numpy as np
from scipy import stats

# ---------- 1. Friedman 检验（多算法排名） ----------
def friedman_test(results_matrix):
    \"\"\"
    results_matrix: shape = (n_targets, n_algorithms)
    每行是一个target，每列是一个算法在该target上的性能值
    返回: statistic, p_value
    \"\"\"
    from scipy.stats import friedmanchisquare

    # 将结果转换为排名（每行内排名，值越大排名越高）
    ranks = np.argsort(np.argsort(-results_matrix, axis=1), axis=1) + 1

    # Friedman检验
    stat, p = friedmanchisquare(*[results_matrix[:, i] for i in range(results_matrix.shape[1])])
    return stat, p

# ---------- 2. Wilcoxon 符号秩检验（两两对比） ----------
def wilcoxon_pairwise(results_matrix, algorithm_names):
    \"\"\"
    对每对算法做Wilcoxon检验
    返回: 字典 {(alg_i, alg_j): (statistic, p_value, effect_size)}
    \"\"\"
    from scipy.stats import wilcoxon
    n_algs = results_matrix.shape[1]
    results = {}

    for i in range(n_algs):
        for j in range(i+1, n_algs):
            # 计算差值
            diff = results_matrix[:, i] - results_matrix[:, j]
            # 只保留非零差值
            nonzero = diff != 0
            if np.sum(nonzero) < 5:
                continue

            stat, p = wilcoxon(diff[nonzero])

            # 计算效应量 (r = Z / sqrt(N))
            z = stats.norm.ppf(1 - p/2) if p < 1 else 0
            n = np.sum(nonzero)
            r = z / np.sqrt(n) if n > 0 else 0

            results[(algorithm_names[i], algorithm_names[j])] = {
                'statistic': stat,
                'p_value': p,
                'effect_size_r': r,
                'significant': p < 0.05
            }

    return results

# ---------- 3. Nemenyi 事后检验 ----------
def nemenyi_test(results_matrix, algorithm_names, alpha=0.05):
    \"\"\"
    Nemenyi post-hoc test for Friedman
    判断哪些算法对之间有显著差异
    \"\"\"
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
    \"\"\"
    计算Cohen's d效应量
    d < 0.2: negligible
    0.2 <= d < 0.5: small
    0.5 <= d < 0.8: medium
    d >= 0.8: large
    \"\"\"
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

# ---------- 5. 跨层分析 ----------
def cross_layer_analysis(results_by_layer, algorithm_names):
    \"\"\"
    分析算法在不同营养素维度层的表现差异
    results_by_layer: {'low': results_matrix_low, 'medium': ..., 'high': ...}
    \"\"\"
    from scipy.stats import kruskal

    print("\\n【跨层性能分析】")
    for layer, matrix in results_by_layer.items():
        mean_perf = np.mean(matrix, axis=0)
        std_perf = np.std(matrix, axis=0)
        print(f"\\n{layer}层 (n={matrix.shape[0]} targets):")
        for i, (m, s) in enumerate(zip(mean_perf, std_perf)):
            print(f"  {algorithm_names[i]}: mean={m:.4f} +/- {s:.4f}")

    # Kruskal-Wallis检验（跨层比较同一算法）
    n_algs = list(results_by_layer.values())[0].shape[1]

    print(f"\\n【Kruskal-Wallis 跨层检验】")
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
"""

# 保存统计检验框架到文件
def save_statistical_tests(output_path: str = 'statistical_tests_framework.py'):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(STATISTICAL_TESTS_CODE)
    print(f"统计检验框架已保存到: {output_path}")

# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    CSV_PATH = 'target.csv'  # 修改为你的数据路径
    OUTPUT_PATH = 'data/representative_targets_100.csv'

    reps = main(
        csv_path=CSV_PATH,
        n_representatives=100,
        output_path=OUTPUT_PATH
    )

    # 保存统计检验框架
    save_statistical_tests('statistical_tests_framework.py')
