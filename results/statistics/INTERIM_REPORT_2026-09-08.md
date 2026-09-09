# NutriOpt-Bench 阶段性结果报告（Interim）

**日期：** 2026-09-08  
**定位：** RCFA / IEEE TEVC survey+benchmark（优化机制对比，非临床营养评价）  
**证据等级约定：** 仅 `PRIMARY` / `PRIMARY_LITE`（须标注） / `EXACT` / `ROBUSTNESS` 可支撑科学表述；`PROFILE`/`PILOT`/`DEBUG` 不作排名依据。

---

## 1. 执行快照

| 实验 | 协议 | 状态 | 进度（约） | 产物 |
|------|------|------|-----------|------|
| **E2 PRIMARY_LITE** | NHANES-10K；12 tgt × 5 seed × 10 algo × 3 sparsity；pop=200；20k evals | **完成** | 1800/1800 | `results/raw/e2_sparsity_primary_lite.csv` |
| **E1 PRIMARY** | 8 scales × ≤12 tgt × 10 seed × 10 algo；pop=500；100k；**24 workers** | **进行中** | ~590/8400（Foods.com 段） | `e1_scaling_full.csv.partial.csv` |
| **E3 PRIMARY** | NHANES-10K；dims {6,9,12,16}；12×10×8；100k；**24 workers** | **进行中** | ~400/3840（仍在 dim=6） | `e3_dimension_full.csv.partial.csv` |
| **E2 full PRIMARY** | 同 lite，但 100k × 10 seeds；workers=32 | **排队**（E1+E3 结束后自动启动） | — | `watch_then_e2_full.py` |
| Exact MILP | NHANES-500；cand_limit=300；12 reps | **完成** | 12/12 | `results/raw/exact_small.csv` |
| Target coverage | 4019 vs cluster/random | **完成** | — | `results/target_coverage.csv` |

进程池已启用；E1/E3 并行后墙钟 ETA 约数小时量级（大尺度 NHANES 仍会拉长）。遗留 `e1_scaling.csv` = **DEBUG**，禁止引用。

---

## 2. 已完成：可引用结论

### 2.1 Exact MILP 校准（EXACT）

在缩小候选池（500 食物、候选上限 300）上，CBC 对 12 个代表目标均返回 `Optimal`，但 **4/12 仍营养不可行**（正松弛）：

| 结果 | 目标 |
|------|------|
| 可行 | rep_001–004, 006–007, 009–010 |
| 不可行 | rep_000, 005, 008, 011 |

**含义：** FHR=0 可能来自**实例本身无可行解**，不能直接判算法失败。

### 2.2 代表目标覆盖（描述性）

标准化特征空间中，cluster-12 的 mean NN 距离 **3.77** < random-12 的 **4.54**（覆盖代理更好）。见 `results/figures/fig_target_coverage.png`。

### 2.3 E2 PRIMARY_LITE — 稀疏度主效应（核心完成结果）

**设置：** NHANES-10K；稀疏因子 \(f\in\{0.5,1.0,1.5\}\)；墙钟 ≈3.1 h。  
**标签：** PRIMARY_LITE ≠ full PRIMARY（非 100k×10）。

#### 平均 FHR（12×5 聚合）

| \(f\) | ACMO | BMOEA | MA | NSGA2 | SPEA2 | RVEA | CTAEA | Greedy | Repair | Random |
|------:|-----:|------:|---:|------:|------:|-----:|------:|-------:|-------:|-------:|
| 0.5 | 0.15 | 0.15 | **0.18** | 0.17 | 0.13 | 0.13 | 0.08 | 0.05 | 0.10 | **0.02** |
| 1.0 | 0.82 | 0.75 | **0.92** | 0.72 | 0.63 | 0.50 | 0.40 | 0.33 | 0.40 | 0.20 |
| 1.5 | 0.98 | 0.97 | **1.00** | **1.00** | 0.93 | 0.87 | 0.88 | 0.82 | 0.97 | 0.63 |

- **主效应：** 所有算法 FHR 随区间放宽单调上升（紧约束难、松约束易）。  
- **整体量级：** 约 0.5→0.12，1.0→0.57，1.5→0.91（全算法平均，见报告 `E2_PRIMARY_LITE_REPORT.md`）。

#### Friedman（按目标聚合 FHR）

| \(f\) | 统计量 | \(p\) |
|------:|-------:|------:|
| 0.5 | 18.20 | 0.033 |
| 1.0 | 67.12 | \(5.6\times10^{-11}\) |
| 1.5 | 35.00 | \(6.0\times10^{-5}\) |

三档均拒绝“算法表现相同”。

#### 秩稳定性（跨稀疏度；1=最好）

| 算法 | 平均秩 | 秩极差（跨 \(f\)） |
|------|------:|------------------:|
| ACMO-Diet | **1.39** | **0.42**（最稳居前） |
| BMOEA | 2.72 | 0.83 |
| MA | 3.81 | 1.50 |
| SPEA2 | 7.94 | **3.00**（变动最大） |
| CTAEA | 4.89 | 2.67 |
| Random | 8.69 | 0.58 |

**暂读：** 可行性偏向 / memetic 族在紧–松区间上更稳居前；部分 archive/indicator MOEA 秩随稀疏度漂移 → **算法×稀疏度交互** 的提示性证据（须 full PRIMARY 再定论）。

图：`results/figures/fig_e2_primary_lite_fhr_sparsity.png`，`..._rank_sparsity.png`。

### 2.4 公平性披露（E2 LITE）

| 算法 | median `n_evals` | 说明 |
|------|-----------------:|------|
| 多数 EA / Random | 20000 | 打满预算 |
| MA / Greedy | ~20130–20320 | 轻微超支（内层评分计入） |
| **RandomRepair** | **~3999** | **严重吃不满预算**（已修代码；本表不可硬排名 Repair） |

---

## 3. 进行中：仅作工程进度，不作 PRIMARY 结论

### 3.1 E1 partial（~550 行，几乎全是 Foods.com）

当前 Foods.com 上各算法 mean FHR≈1.0（小菜谱库、4 营养素、易可行）。  
**尚未进入** NHANES 5K–full 缩放轴 → **不能**据此谈“规模效应”或算法排序。

### 3.2 E3 partial（~400 行，仅 dim=6）

dim=6 上多数算法 FHR≈1.0（CTAEA≈0.98）。  
**9/12/16 维未出** → **不能**谈维度诅咒。

---

## 4. 明确禁止的表述（当前）

1. “算法 X 全面最优 / SOTA”。  
2. 用 legacy `e1_scaling.csv` 或 PROFILE/PILOT 作排名。  
3. 把 E2 PRIMARY_LITE 写成 full PRIMARY（100k×10）。  
4. 在 Repair 未等预算重跑前，用 LITE 表贬低/抬高 RandomRepair。  
5. 任何临床有效性 / 膳食充分性宣称。

---

## 5. 下一步

1. 等 E1、E3 PRIMARY 写完终表 → 自动启动 E2 full（32 workers）。  
2. 对 E1（按 dataset 分层）、E3（按 dim）、E2 full 跑 Friedman / 秩稳定性 /（可选）交互模型。  
3. 用等预算 Repair 重跑后更新机制对比表。  
4. 把手稿 Results 从 “PRIMARY_LITE + pending” 换成完整 PRIMARY 表。

---

## 6. 关键路径速查

| 类型 | 路径 |
|------|------|
| 本报告 | `results/statistics/INTERIM_REPORT_2026-09-08.md` |
| E2 LITE 详报 | `results/statistics/E2_PRIMARY_LITE_REPORT.md` |
| E2 原始 / 表 / 图 | `results/raw/e2_sparsity_primary_lite.csv`，`results/tables/e2_primary_lite_*`，`results/figures/fig_e2_primary_lite_*` |
| MILP | `results/raw/exact_small.csv` |
| 手稿 | `manuscript/nutriopt_bench.pdf` |
| 进度日志 | `results/logs/e1_full.log`，`e3_full.log`，`watch_then_e2_full.log` |

---

*报告生成时 E1≈590/8400、E3≈400/3840；数字以磁盘 CSV / 日志最新一行为准。*
