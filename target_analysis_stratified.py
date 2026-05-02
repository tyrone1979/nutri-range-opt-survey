#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target 分析与分层代表性抽取系统
功能：
1. 全量 target 的营养成分维度分析
2. 按营养素维度数分层（低6种 / 中7-9种 / 高10+种）——基于实际数据分布
3. 每层内部聚类抽取代表性 target
4. 确保统计显著性（优先 n>=30，允许 n>=20）
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

# 16种营养成分字段
NUTRIENT_FIELDS = [
    'calorie', 'protein', 'carb', 'sugar', 'saturated_fat', 'cholesterol',
    'sodium', 'potassium', 'fiber', 'calcium', 'vitamin_d', 'vitamin_c',
    'iron', 'folic_acid', 'vitamin_b12', 'phosphorus'
]

def analyze_dimensions(targets: List[Dict]) -> Dict:
    """
    分析每个target的营养素维度数，并分层
    基于实际数据分布：最小维度为6，因此分层为：
    - low: 6种（约6.5%）
    - medium: 7-9种（约71.3%）
    - high: 10+种（约22.1%）
    """
    dim_stats = {'low': [], 'medium': [], 'high': []}

    for idx, t in enumerate(targets):
        n_dims = len(t.keys())
        if n_dims <= 6:  # 低维度
            dim_stats['low'].append(idx)
        elif n_dims <= 9:  # 中维度
            dim_stats['medium'].append(idx)
        else:  # 高维度
            dim_stats['high'].append(idx)

    return dim_stats

def print_dimension_report(targets: List[Dict], dim_stats: Dict):
    """打印维度分层报告"""
    total = len(targets)
    print("=" * 70)
    print("【营养素维度分层统计】")
    print("=" * 70)

    for layer_name, layer_label, idx_list in [
        ('low', '低维度 (<=6种)', dim_stats['low']),
        ('medium', '中维度 (7-9种)', dim_stats['medium']),
        ('high', '高维度 (10+种)', dim_stats['high']),
    ]:
        n = len(idx_list)
        print(f"\n{layer_label}: {n} 个 ({n/total*100:.1f}%)")

        field_counts = Counter()
        for idx in idx_list:
            field_counts.update(targets[idx].keys())

        print("  字段分布:")
        for field, cnt in field_counts.most_common():
            print(f"    {field:<15}: {cnt:4d} ({cnt/n*100:5.1f}%)")

# ============================================================
# 3. 特征工程
# ============================================================

def build_feature_matrix(targets: List[Dict], field_list: List[str], stats: Dict) -> np.ndarray:
    """
    将target列表转为特征矩阵
    每维提取 [lower, upper]，缺失用全局中位数填充
    9999999.0 无限上限用合理值替代
    """
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
    return np.array(X)

def compute_global_stats(targets: List[Dict], field_list: List[str]) -> Dict:
    """计算全局统计量用于缺失值填充"""
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
    return stats

# ============================================================
# 4. 特征标签生成
# ============================================================

def generate_feature_tags(t: Dict) -> List[str]:
    """为target生成可读的业务特征标签"""
    tags = []

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

    if 'protein' in t:
        pro = t['protein']
        if pro[1] > 100:
            tags.append("高蛋白")
        elif pro[1] < 70:
            tags.append("低蛋白")
        if pro[0] > 50:
            tags.append(f"高下限蛋白{pro[0]}+")

    if 'carb' in t:
        carb = t['carb']
        if carb[1] > 600:
            tags.append("高碳水")
        if carb[0] > 300:
            tags.append(f"高下限碳水{carb[0]}+")
        elif carb[0] == 0:
            tags.append("碳水无下限")

    if 'sugar' in t:
        tags.append("严格控糖<=25" if t['sugar'][1] <= 25 else "控糖<=36")
    else:
        tags.append("无糖限制")

    if 'saturated_fat' in t:
        sf = t['saturated_fat']
        if sf[1] >= 99999:
            tags.append("SatFat无上限")
        elif sf[1] <= 20:
            tags.append("低SatFat<=20")
        else:
            tags.append(f"SatFat<={sf[1]:.0f}")

    if 'cholesterol' in t and t['cholesterol'][1] <= 200:
        tags.append("低胆固醇<=200")

    if 'fiber' in t:
        tags.append(f"纤维>={t['fiber'][0]}")

    if 'sodium' in t:
        tags.append("限钠")
    if 'potassium' in t:
        tags.append("高钾")

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
# 5. 分层聚类与代表性抽取
# ============================================================

@dataclass
class Representative:
    rep_id: int
    layer: str          # 'low'/'medium'/'high'
    cluster_id: int
    original_idx: int
    size: int
    strength: str       # 'strong'/'moderate'/'acceptable'/'weak'
    features: List[str]
    target: Dict

def cluster_and_extract(
    targets: List[Dict],
    indices: List[int],
    layer_name: str,
    field_list: List[str],
    global_stats: Dict,
    n_clusters: int,
    min_cluster_size: int = 15
) -> List[Representative]:
    """
    对指定层进行聚类并抽取代表性target
    使用更宽松的min_cluster_size以产生更多有效簇
    """
    layer_targets = [targets[i] for i in indices]
    n_samples = len(layer_targets)

    if n_samples < n_clusters:
        n_clusters = max(2, n_samples // min_cluster_size)

    # 构建特征矩阵
    X_local = build_feature_matrix(layer_targets, field_list, global_stats)
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

    # 小簇合并
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

    for cid in final_ids:
        cluster_indices_local = np.where(new_labels == cid)[0]
        cluster_indices_global = [indices[i] for i in cluster_indices_local]
        cluster_points = X_scaled[cluster_indices_local]

        center = np.mean(cluster_points, axis=0)
        dists = cdist(cluster_points, center.reshape(1, -1), metric='euclidean').flatten()
        closest_local = cluster_indices_local[np.argmin(dists)]
        closest_global = indices[closest_local]

        size = len(cluster_indices_global)
        if size >= 30:
            strength = 'strong'
        elif size >= 25:
            strength = 'moderate'
        elif size >= 20:
            strength = 'acceptable'
        else:
            strength = 'weak'

        t = targets[closest_global]
        features = generate_feature_tags(t)

        reps.append(Representative(
            rep_id=0,  # 稍后统一分配
            layer=layer_name,
            cluster_id=cid,
            original_idx=closest_global,
            size=size,
            strength=strength,
            features=features,
            target=t
        ))

    return reps

# ============================================================
# 6. 分层配额计算
# ============================================================

def calculate_quotas(n_total: int, dim_stats: Dict, min_per_layer: int = 5) -> Dict:
    """
    按比例分配每层抽取数，确保每层至少 min_per_layer 个
    实际数据分布: low约6%, medium约71%, high约22%
    """
    total_samples = sum(len(v) for v in dim_stats.values())

    # 先按样本比例分配
    raw_quotas = {}
    for layer in ['low', 'medium', 'high']:
        raw_quotas[layer] = int(n_total * len(dim_stats[layer]) / total_samples)

    # 保底
    quotas = {}
    remaining = n_total

    for layer in ['low', 'medium', 'high']:
        quotas[layer] = max(min_per_layer, raw_quotas[layer])
        remaining -= quotas[layer]

    # 余量按样本量比例分配给中/高层
    if remaining > 0:
        for layer in ['medium', 'high', 'low']:
            extra = min(remaining, len(dim_stats[layer]) // 30)
            quotas[layer] += extra
            remaining -= extra
            if remaining <= 0:
                break

    return quotas

# ============================================================
# 7. 主流程
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

    # 3. 计算全局统计
    global_stats = compute_global_stats(targets, NUTRIENT_FIELDS)

    # 4. 计算配额
    print("\n" + "=" * 70)
    print("【Step 3】计算分层配额")
    quotas = calculate_quotas(n_representatives, dim_stats, min_per_layer=5)
    for layer, q in quotas.items():
        print(f"  {layer}层: 配额 {q} 个代表（支撑: {len(dim_stats[layer])}样本）")

    # 5. 分层聚类抽取
    print("\n" + "=" * 70)
    print("【Step 4】分层聚类抽取")

    all_reps = []
    for layer, quota in quotas.items():
        if quota <= 0 or len(dim_stats[layer]) == 0:
            continue

        n_samples = len(dim_stats[layer])
        n_clusters = min(quota * 3, max(2, n_samples // 20))

        print(f"\n>>> {layer}层: {n_samples}样本 -> 聚类{n_clusters} -> 计划抽取{quota}个")

        reps = cluster_and_extract(
            targets=targets,
            indices=dim_stats[layer],
            layer_name=layer,
            field_list=NUTRIENT_FIELDS,
            global_stats=global_stats,
            n_clusters=n_clusters,
            min_cluster_size=15
        )

        # 按簇大小排序，选前quota个
        reps_sorted = sorted(reps, key=lambda x: x.size, reverse=True)
        selected = reps_sorted[:quota]

        print(f"  实际生成 {len(reps)} 个簇，选取 {len(selected)} 个")

        all_reps.extend(selected)

    # 6. 全局排序，确保最终数量
    all_reps_sorted = sorted(all_reps, key=lambda x: x.size, reverse=True)
    final_reps = all_reps_sorted[:n_representatives]

    # 更新全局ID
    for i, r in enumerate(final_reps):
        r.rep_id = i

    # 7. 统计报告
    print("\n" + "=" * 70)
    print("【Step 5】统计验证")

    total_covered = sum(r.size for r in final_reps)
    strong = len([r for r in final_reps if r.strength == 'strong'])
    moderate = len([r for r in final_reps if r.strength == 'moderate'])
    acceptable = len([r for r in final_reps if r.strength == 'acceptable'])
    weak = len([r for r in final_reps if r.strength == 'weak'])

    print(f"  最终代表数: {len(final_reps)}")
    print(f"  强统计(>=30): {strong} ({strong/len(final_reps)*100:.1f}%)")
    print(f"  中统计(25-29): {moderate} ({moderate/len(final_reps)*100:.1f}%)")
    print(f"  可接受(20-24): {acceptable} ({acceptable/len(final_reps)*100:.1f}%)")
    print(f"  弱统计(<20): {weak} ({weak/len(final_reps)*100:.1f}%)")
    print(f"  覆盖样本: {total_covered}/{len(targets)} ({total_covered/len(targets)*100:.1f}%)")

    # 按层统计
    for layer in ['low', 'medium', 'high']:
        layer_reps = [r for r in final_reps if r.layer == layer]
        if layer_reps:
            dims = [len(r.target.keys()) for r in layer_reps]
            print(f"\n  {layer}层 ({len(layer_reps)}个代表):")
            print(f"    维度范围: [{min(dims)}-{max(dims)}], 平均 {np.mean(dims):.1f}")
            for r in layer_reps:
                print(f"    [{r.rep_id:3d}] {r.strength:10s} size={r.size:3d} | {' | '.join(r.features[:5])}")

    # 8. 保存
    if output_path:
        print("\n" + "=" * 70)
        print("【Step 6】保存结果")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rep_id', 'layer', 'cluster_id', 'original_idx', 'cluster_size',
                'statistical_strength', 'tags', 'target_json'
            ])

            for r in final_reps:
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

    return final_reps

# ============================================================
# 8. 入口
# ============================================================

if __name__ == '__main__':
    CSV_PATH = 'target.csv'  # 修改为你的数据路径
    OUTPUT_PATH = 'data/representative_targets_100_stratified.csv'

    reps = main(
        csv_path=CSV_PATH,
        n_representatives=100,
        output_path=OUTPUT_PATH
    )
