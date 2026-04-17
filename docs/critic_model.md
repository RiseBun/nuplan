# NuPlan Critic 模型文档

## 1. 概述

NuPlan Critic 是一个**轨迹质量评价模型**，用于判断候选规划轨迹在当前驾驶场景下是否合理。模型以二分类方式工作：给定当前场景的历史图像、车辆状态和一条候选轨迹，输出该轨迹为"好轨迹"的概率。

**核心用途**：作为端到端规划系统的 Critic/打分模块，对规划器生成的多条候选轨迹进行筛选和排序。

## 2. 模型架构

CriticModel 由三个编码器和一个融合分类头组成：

```
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ SimpleImageEncoder│  │ TrajectoryEncoder │  │   EgoEncoder     │
│  (4层 CNN + Pool) │  │  (2层 MLP)        │  │  (2层 MLP)       │
│  输入: 历史图像    │  │  输入: 候选轨迹    │  │  输入: 车辆状态   │
│  输出: 256-d       │  │  输出: 128-d       │  │  输出: 128-d     │
└────────┬──────────┘  └────────┬──────────┘  └────────┬────────┘
         │                      │                      │
         └──────────────┬───────┴──────────────────────┘
                        │ Concat (512-d)
                  ┌─────┴─────┐
                  │ FusionHead │
                  │ (3层 MLP)  │
                  │  → 1-d     │
                  └─────┬─────┘
                        │
                    logit (标量)
                  sigmoid → 概率
```

### 模型参数

| 组件 | 参数 |
|------|------|
| image_feature_dim | 256 |
| action_feature_dim | 128 |
| hidden_dim | 256 |
| dropout | 0.1 |
| 模型大小 | ~7.8 MB |

## 3. 输入与输出

### 3.1 模型输入

| 输入 | Shape | 说明 |
|------|-------|------|
| `images` | `(B, 4, 3, 224, 224)` | 最近 4 帧前视相机图像 (CAM_F0)，ImageNet 归一化 |
| `ego_state` | `(B, 5)` | 车辆状态：`[vx, vy, yaw_rate, ax, angular_rate_z]`，tanh 归一化 |
| `candidate_traj` | `(B, 8, 3)` | 候选轨迹：8 个时间步 × `(dx, dy, dyaw)`，tanh 归一化 |

- 时间步间隔 `future_step_time_s = 0.5s`，共覆盖未来 4 秒
- 轨迹坐标为相对于当前车辆位姿的局部坐标

### 3.2 模型输出

| 输出 | Shape | 说明 |
|------|-------|------|
| `logit` | `(B,)` | 未经 sigmoid 的原始分数 |
| `sigmoid(logit)` | `(B,)` | 轨迹为"好轨迹"的概率，阈值 0.5 |

### 3.3 训练数据格式

训练数据为 JSONL 格式索引文件，每行一个样本：

```json
{
  "sample_id": "scene_name__timestamp__pos",
  "scene_name": "2021.05.12.22.00.38_veh-35_01008_01518",
  "timestamp_us": 1620857889987476,
  "history_images": [
    "nuplan-v1.1_mini_camera_0/scene_name/CAM_F0/ed564f55883d50d3.jpg",
    "nuplan-v1.1_mini_camera_0/scene_name/CAM_F0/51a6be8a692758aa.jpg",
    "nuplan-v1.1_mini_camera_0/scene_name/CAM_F0/eb2654f04a76556b.jpg",
    "nuplan-v1.1_mini_camera_0/scene_name/CAM_F0/bdd2f54ea521590c.jpg"
  ],
  "ego_state": [3.96, -0.07, -2.04, -1.35, -0.002],
  "candidate_traj": [[1.78, -0.01, -0.001], ...],
  "label": 1
}
```

- `label=1`：正样本（真实未来轨迹）
- `label=0`：负样本（从其他时间点采样的轨迹）
- 正负样本比例 1:1

## 4. 数据流水线

```
NuPlan SQLite (.db)  ──┐
                       ├── build_critic_index.py ──→ critic_train.jsonl
相机图像 (CPFS)       ──┘                           critic_val.jsonl
                                                        │
                                                        ▼
                                                    train.py
                                                        │
                                                        ▼
                                                checkpoints/best.pth
                                                        │
                                                        ▼
                                                eval_critic.py
```

### 4.1 索引构建

```bash
python tools/build_critic_index.py \
    --db-root /mnt/datasets/e2e-datasets/20260227/e2e-datasets/dataset_pkgs/nuplan-v1.1/splits/mini \
    --image-roots /mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set/nuplan-v1.1_mini_camera_0 \
                  /mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set/nuplan-v1.1_mini_camera_1 \
    --output-dir indices/
```

当前数据规模：

| 划分 | 场景数 | 样本数 |
|------|--------|--------|
| train | 11 | 19,118 |
| val | 3 | 5,192 |

## 5. 用法

### 5.1 本地训练（单机）

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan
python train.py --config configs/train_critic_mini.py
```

可选参数：

```bash
python train.py --config configs/train_critic_mini.py \
    --epochs 20 \
    --batch-size 16 \
    --work-dir ./work_dirs/my_experiment
```

### 5.2 DLC 分布式训练

**Volume 配置**（必须）：

```json
{
  "volumes": [
    {
      "name": "cpfs-ares-root",
      "mountPath": "/mnt/cpfs/"
    }
  ]
}
```

**启动命令**：

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh
```

`dlc_train.sh` 自动处理多机多卡的 torchrun 配置，支持透传参数：

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh \
    --batch-size 16 --epochs 20
```

### 5.3 评估

```bash
# 在验证集上评估 best checkpoint
python eval_critic.py --checkpoint work_dirs/critic_mini_v1/checkpoints/best.pth

# 在训练集上评估
python eval_critic.py --checkpoint work_dirs/critic_mini_v1/checkpoints/best.pth --split train

# 限制评估样本数（快速验证）
python eval_critic.py --checkpoint work_dirs/critic_mini_v1/checkpoints/best.pth --max-samples 200
```

## 6. 评估指标说明

### 6.1 分类指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **Accuracy** | `(TP+TN) / (TP+TN+FP+FN)` | 整体分类准确率 |
| **Precision** | `TP / (TP+FP)` | 预测为正样本中真正为正的比例，高 Precision 意味着少误判 |
| **Recall** | `TP / (TP+FN)` | 所有正样本中被正确识别的比例，高 Recall 意味着少漏判 |
| **F1 Score** | `2×P×R / (P+R)` | Precision 和 Recall 的调和平均，综合衡量分类质量 |

### 6.2 混淆矩阵

|  | 预测=好轨迹 | 预测=差轨迹 |
|--|------------|------------|
| 实际=好轨迹 | **TP** (正确放行) | **FN** (错误拒绝) |
| 实际=差轨迹 | **FP** (错误放行) | **TN** (正确拒绝) |

### 6.3 概率分布指标

| 指标 | 含义 |
|------|------|
| `pos_prob_mean` | 正样本的平均预测概率，理想值接近 1.0 |
| `neg_prob_mean` | 负样本的平均预测概率，理想值接近 0.0 |

两者差距越大，模型区分能力越强。

### 6.4 当前基线结果（mini_set, epoch=10）

| 指标 | 值 |
|------|-----|
| Accuracy | 77.3% |
| Precision | 70.4% |
| Recall | 94.1% |
| F1 Score | 80.6% |
| Val Loss | 0.454 |
| pos_prob_mean | 0.658 |
| neg_prob_mean | 0.292 |

**分析**：模型召回率高（94%），很少漏掉好轨迹；但 Precision 偏低（70%），部分差轨迹被误判为好轨迹。后续可通过增加数据量、调整正负样本比例或增大模型容量来改善。

## 7. 项目结构

```
nuplan/
├── configs/
│   └── train_critic_mini.py      # 训练配置
├── indices/
│   ├── critic_train.jsonl        # 训练索引 (19,118 条)
│   ├── critic_val.jsonl          # 验证索引 (5,192 条)
│   └── critic_index_summary.json # 索引统计信息
├── scripts/
│   └── dlc_train.sh              # DLC 分布式训练脚本
├── tools/
│   └── build_critic_index.py     # 索引构建工具
├── train.py                      # 训练入口（含模型定义和数据集）
├── eval_critic.py                # 评估脚本
└── work_dirs/
    └── critic_mini_v1/
        ├── checkpoints/
        │   ├── best.pth          # 最佳验证 loss 的 checkpoint
        │   └── latest.pth        # 最新 epoch 的 checkpoint
        ├── config_snapshot.json   # 训练时的配置快照
        └── eval_val_results.json # 评估结果
```

## 8. 训练配置参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 10 | 训练轮数 |
| batch_size | 8 | 每 GPU 的 batch size |
| lr | 1e-4 | 学习率 |
| weight_decay | 1e-2 | 权重衰减 |
| image_size | 224 | 输入图像尺寸 |
| history_num_frames | 4 | 历史帧数 |
| candidate_traj_steps | 8 | 候选轨迹时间步数 |
| positive_weight | 1.0 | BCEWithLogitsLoss 正样本权重 |
| log_interval | 20 | 日志打印间隔 (steps) |
| save_interval | 1 | Checkpoint 保存间隔 (epochs) |
