# NuPlan Consistency Critic 模型文档 (v2)

> 第二代 Critic 模型，在[第一代 Critic](critic_model.md) 基础上引入**双头评估**和**未来图像输入**。

## 1. 概述

Consistency Critic 是第二代轨迹质量评价模型，核心改进：

1. **新增未来图像输入**：除历史 4 帧外，额外输入未来 4 帧图像（0.5/1.0/1.5/2.0s），让模型能对比"轨迹预期"与"实际发生"
2. **双头输出**：分别评估 Consistency（轨迹-场景一致性）和 Validity（轨迹合理性）
3. **多样化负样本**：从单一轨迹交换扩展为 3 类 6 种负样本，覆盖更多失败模式

**核心用途**：作为端到端规划系统的 Critic/打分模块，Validity 粗筛不合理轨迹，Consistency 精选最匹配场景的轨迹。

### 1.1 与第一代的对比

| 维度 | v1 Critic | v2 Consistency Critic |
|------|-----------|----------------------|
| 输入图像 | 历史 4 帧 | 历史 4 帧 + **未来 4 帧** |
| 输出头 | 1 个（好/坏轨迹） | **2 个**（Consistency + Validity） |
| 负样本类型 | 轨迹交换（1 种） | 轨迹交换 + 图像交换 + 3 种扰动（**5 种**） |
| 正负比 | 1:1 | 1:3（每个正样本对应 3 个负样本） |
| 参数量 | ~0.84M | ~0.84M |
| 训练数据 | 19,118 / 5,192 | 38,236 / 10,384 |

## 2. 双头任务定义

### 2.1 Consistency Head（一致性头）

**判断轨迹是否匹配当前场景的真实未来演进。**

- 输入同时包含历史图像和未来图像
- 模型需要判断：候选轨迹是否**真正对应**未来图像所呈现的真实行驶结果
- 正样本：GT 轨迹 + 对应的真实未来图像（一一匹配）
- 负样本：轨迹与未来场景之间存在不匹配

应用场景：在线轨迹对齐检测、后验评估。

### 2.2 Validity Head（有效性头）

**判断轨迹本身是否物理/语义合理，不要求与真实未来完全匹配。**

- 关键区别：`image_swap`（替换未来图像）时，轨迹本身仍是某个场景的 GT 轨迹，仍标为**有效**
- 正样本：任何来自真实场景的 GT 轨迹（无论图像是否匹配）
- 负样本：经过扰动或交换后不再合理的轨迹

应用场景：轨迹安全筛选、异常轨迹过滤。

### 2.3 标签矩阵

| 数据源 | 含义 | Consistency | Validity |
|--------|------|:-----------:|:--------:|
| `gt_pos` | GT 轨迹 + 真实图像 | 1 | 1 |
| `image_swap` | GT 轨迹 + **另一时刻**的未来图像 | 0 | **1** |
| `traj_swap` | **另一时刻**的轨迹 + 真实图像 | 0 | 0 |
| `perturb_lateral` | GT 轨迹 + 横向偏移 (0.5~2.0m) | 0 | 0 |
| `perturb_heading` | GT 轨迹 + 航向扰动 (5~15 度) | 0 | 0 |
| `perturb_speed` | GT 轨迹 + 速度缩放 (0.7~1.3x) | 0 | 0 |

## 3. 模型架构

```
┌──────────────────┐  ┌──────────────────┐
│  History Images   │  │  Future Images    │
│  (B,4,3,224,224)  │  │  (B,4,3,224,224)  │
└────────┬─────────┘  └────────┬──────────┘
         │                     │
   ┌─────┴─────┐        ┌─────┴─────┐
   │ SharedCNN  │        │ SharedCNN  │     ← 共享 4 层 CNN Backbone
   │ + HistProj │        │ + FutProj  │
   └─────┬─────┘        └─────┬─────┘
         │                     │
      z_hist (256)          z_fut (256)
         │                     │
         └──────────┬──────────┘
                    │
┌───────────────┐   │   ┌──────────────┐
│ TrajEncoder   │   │   │  EgoEncoder  │
│ (B,8,3)→128-d │   │   │ (B,5)→128-d  │
└───────┬───────┘   │   └──────┬───────┘
        │           │          │
        └─────┬─────┴──────────┘
              │ Concat (768-d)
        ┌─────┴──────┐
        │SharedFusion│
        │ 768→256→256│
        │ +Dropout   │
        └─────┬──────┘
              │ (256-d)
        ┌─────┴──────┐
        │            │
  ┌─────┴─────┐ ┌───┴──────┐
  │Consistency│ │ Validity │
  │Head (→1)  │ │Head (→1) │
  └───────────┘ └──────────┘
```

### 模型参数

| 组件 | 维度 | 参数量 |
|------|------|--------|
| SharedCNN Backbone | 3→32→64→128→256 | ~300K |
| HistoryProj + FutureProj | 256→256 × 2 | ~131K |
| TrajectoryEncoder | 24→256→128 | ~39K |
| EgoEncoder | 5→128→128 | ~17K |
| SharedFusion | 768→256→256 | ~263K |
| ConsistencyHead + ValidityHead | 256→1 × 2 | ~0.5K |
| **总计** | | **~0.84M** |

## 4. 输入与输出

### 4.1 模型输入

| 输入 | Shape | 说明 |
|------|-------|------|
| `history_images` | `(B, 4, 3, 224, 224)` | 最近 4 帧前视相机图像 (CAM_F0)，ImageNet 归一化 |
| `future_images` | `(B, 4, 3, 224, 224)` | 未来 4 帧图像 (t+0.5s, +1.0s, +1.5s, +2.0s) |
| `ego_state` | `(B, 5)` | 车辆状态：`[vx, vy, yaw, ax, angular_rate_z]` |
| `candidate_traj` | `(B, 8, 3)` | 候选轨迹：8 步 × `(dx, dy, dyaw)`，线性归一化 |

- 轨迹时间步 0.5s，共覆盖未来 4 秒
- 轨迹线性归一化缩放因子：`dx/60.0, dy/25.0, dyaw/2.0`

### 4.2 模型输出

| 输出 | Shape | 说明 |
|------|-------|------|
| `consistency_logit` | `(B,)` | 一致性原始分数，sigmoid 后 >0.5 为一致 |
| `validity_logit` | `(B,)` | 有效性原始分数，sigmoid 后 >0.5 为有效 |

## 5. 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 30 | 训练轮数 |
| batch_size | 8 | 每 GPU 的 batch size |
| lr | 1e-4 | 学习率 |
| weight_decay | 1e-2 | 权重衰减 |
| lambda_consistency | 1.0 | Consistency 损失权重 |
| lambda_validity | 0.5 | Validity 损失权重 |
| consistency_positive_weight | 3.0 | Consistency BCELoss 正样本权重（正负比 1:3） |
| validity_positive_weight | 1.0 | Validity BCELoss 正样本权重（正负比 1:1） |
| save_interval | 1 | 每 epoch 保存 checkpoint |

损失函数：

```
L = lambda_c * BCE_c(consistency_logit, c_label) + lambda_v * BCE_v(validity_logit, v_label)
```

## 6. 数据流水线

```
NuPlan SQLite (.db)  ──┐
                       ├── build_consistency_index.py ──→ consistency_train.jsonl (38,236 条)
相机图像 (CPFS)       ──┘                                 consistency_val.jsonl  (10,384 条)
                                                              │
                                                              ▼
                                                    train.py (model_type=consistency)
                                                              │
                                                              ▼
                                                  work_dirs/consistency_mini_v1/
                                                      checkpoints/best.pth
                                                              │
                                                              ▼
                                                  eval_critic.py (自动检测 model_type)
```

### 数据规模

| 划分 | 场景数 | Anchor 数 | 总样本数 | 正:负 |
|------|--------|-----------|----------|-------|
| train | 11 | 9,559 | 38,236 | 1:3 |
| val | 3 | 2,596 | 10,384 | 1:3 |

### 验证集样本分布

| 数据源 | 数量 | 占比 |
|--------|------|------|
| gt_pos (正样本) | 2,596 | 25.0% |
| traj_swap | 2,596 | 25.0% |
| image_swap | 2,596 | 25.0% |
| perturb_lateral | 877 | 8.4% |
| perturb_heading | 843 | 8.1% |
| perturb_speed | 876 | 8.4% |

## 7. 评估结果 (mini_set, best epoch=28)

### 7.1 总体指标

| 任务 | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|-----------|--------|-----|-----|
| **Consistency** | 79.0% | 55.0% | 89.3% | 68.0% | 0.873 |
| **Validity** | 81.7% | 75.1% | 94.8% | 83.8% | 0.908 |

### 7.2 各数据源表现

| 数据源 | 数量 | Consistency Acc | Validity Acc | 评价 |
|--------|------|-----------------|--------------|------|
| gt_pos | 2,596 | 89.3% | 90.7% | 正样本召回良好 |
| image_swap | 2,596 | 88.7% | 98.8% | 图像不匹配检测能力强 |
| perturb_heading | 843 | 98.7% | 98.8% | 对航向变化非常敏感 |
| perturb_lateral | 877 | 99.8% | 99.7% | 对横向偏移接近完美 |
| **perturb_speed** | **876** | **13.1%** | **11.0%** | **速度扰动检测失效** |
| traj_swap | 2,596 | 68.0% | 67.7% | 中等，有提升空间 |

### 7.3 问题分析

1. **速度扰动完全失效**（Acc 13%/11%）：模型无法区分速度缩放后的轨迹与正常轨迹。速度缩放仅改变 `dx/dy` 幅值而不改变轨迹形状，当前特征表征对"速度快慢"不敏感。`neg_prob_mean=0.57` 接近正样本水平（0.58），说明模型几乎将所有速度扰动样本视为正样本。

2. **轨迹交换偏弱**（Acc 68%）：来自其他场景的轨迹可能在几何形状上与当前场景相似，模型缺乏足够的语义判别能力。

3. **Consistency Precision 偏低**（55%）：FP=1,898，主要由 perturb_speed（761）和 traj_swap（831）贡献，总计 1,592 占 FP 的 84%。

## 8. 用法

### 8.1 构建索引

```bash
python tools/build_consistency_index.py \
    --db-root /mnt/datasets/.../nuplan-v1.1/splits/mini \
    --image-roots /mnt/cpfs/.../nuplan-v1.1_mini_camera_0 \
                  /mnt/cpfs/.../nuplan-v1.1_mini_camera_1 \
    --output-dir indices/
```

### 8.2 训练

```bash
python train.py --config configs/train_consistency_mini.py
```

### 8.3 评估

```bash
# 评估 best checkpoint（自动检测 model_type=consistency）
python eval_critic.py --checkpoint work_dirs/consistency_mini_v1/checkpoints/best.pth

# 快速评估
python eval_critic.py --checkpoint work_dirs/consistency_mini_v1/checkpoints/best.pth --max-samples 200
```

## 9. 项目结构

```
nuplan/
├── configs/
│   ├── train_critic_mini.py              # v1 训练配置
│   └── train_consistency_mini.py         # v2 训练配置
├── indices/
│   ├── consistency_train.jsonl           # v2 训练索引 (38,236 条)
│   ├── consistency_val.jsonl             # v2 验证索引 (10,384 条)
│   └── consistency_index_summary.json    # 索引统计
├── tools/
│   ├── build_critic_index.py             # v1 索引构建
│   └── build_consistency_index.py        # v2 索引构建
├── train.py                              # 训练入口 (含 v1/v2 模型定义)
├── eval_critic.py                        # 评估脚本 (自动检测 model_type)
└── work_dirs/
    ├── critic_mini_v1/                   # v1 实验目录
    └── consistency_mini_v1/              # v2 实验目录
        ├── checkpoints/
        │   ├── best.pth                  # epoch=28, val_loss=0.6252
        │   └── latest.pth               # epoch=30
        ├── config_snapshot.json
        └── eval_val_results.json
```
