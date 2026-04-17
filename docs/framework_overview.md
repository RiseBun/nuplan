# NuPlan Consistency Critic 框架说明文档

## 📋 项目概述

### 项目定位

**Action-Conditioned Consistency Critic** - 基于动作条件的驾驶场景一致性评估模型

本项目训练一个评分器（Critic），用于评估自动驾驶规划轨迹与视觉场景的一致性，证明其与 nuPlan 闭环性能的相关性优于传统指标（FID/FVD）。

### 核心创新

1. **多维度评估**：6 维度 Action-Conditioned 一致性评分
2. **自动化训练**：World Model 自动生成训练数据
3. **三层验证**：生成质量 → 动作一致性 → 驾驶合理性
4. **实证研究**：证明与闭环性能的相关性

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     训练流程 (Training)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  nuPlan 数据                                                │
│       ↓                                                     │
│  ┌─────────────────────────┐                               │
│  │  DrivingWorld           │  World Model                  │
│  │  (视频生成)             │  生成正负样本对                  │
│  └────────┬────────────────┘                               │
│           ↓                                                 │
│  ┌─────────────────────────┐                               │
│  │  标签计算               │  6 维度自动标注                  │
│  │  (Label Computation)    │  - consistency                │
│  │                         │  - speed/steering/progress    │
│  │                         │  - temporal/validity          │
│  └────────┬────────────────┘                               │
│           ↓                                                 │
│  ┌─────────────────────────┐                               │
│  │  索引构建               │  训练/验证集划分                 │
│  │  (Index Building)       │  类别平衡                       │
│  └────────┬────────────────┘                               │
│           ↓                                                 │
│  ┌─────────────────────────┐                               │
│  │  Consistency Critic     │  多维度多任务学习                │
│  │  模型训练               │  - Multi-head output          │
│  │  (DLC 多卡)             │  - Weighted loss              │
│  └────────┬────────────────┘                               │
│           ↓                                                 │
│      最佳模型 (best.pth)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     评估流程 (Evaluation)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 1: 生成质量                                          │
│  ├─ FID (Fréchet Inception Distance)                       │
│  ├─ FVD (Fréchet Video Distance)                           │
│  └─ LPIPS (Learned Perceptual Image Patch Similarity)      │
│                                                             │
│  Layer 2: Action 一致性 (6 维度)                            │
│  ├─ consistency: 整体一致性 (0/1)                           │
│  ├─ speed_consistency: 速度一致性                           │
│  ├─ steering_consistency: 转向一致性                        │
│  ├─ progress_consistency: 进度一致性                        │
│  ├─ temporal_coherence: 时序连贯性                          │
│  └─ validity: 综合有效性                                    │
│                                                             │
│  Layer 3: 驾驶合理性 (Ranking Metrics)                      │
│  ├─ NDCG@K (Normalized Discounted Cumulative Gain)         │
│  ├─ MRR (Mean Reciprocal Rank)                             │
│  └─ Top-K Hit Rate                                         │
│                                                             │
│  最终验证: Spearman/Kendall 相关性分析                       │
│  Critic 分数 vs nuPlan 闭环性能                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
/mnt/cpfs/prediction/lipeinan/nuplan/
├── 📂 核心训练代码
│   ├── train.py                          # 训练框架（多维度多任务学习）
│   ├── eval_critic.py                    # 评估脚本（Ranking + 相关性）
│   └── evaluate_three_layers.py          # 三层评估整合
│
├── 📂 数据生成与处理
│   ├── generate_critic_training_data.py  # 使用 DrivingWorld 生成训练数据
│   ├── compute_training_labels.py        # 6 维度标签计算
│   └── build_critic_index.py             # 训练索引构建
│
├── 📂 评估器（Layer 1）
│   └── evaluation/
│       ├── fid_calculator.py             # FID 计算
│       ├── fvd_calculator.py             # FVD 计算
│       └── lpips_calculator.py           # LPIPS 计算
│
├── 📂 World Model
│   └── DrivingWorld/                     # 开源视频生成模型
│       └── pretrained_models/
│           ├── world_model.pth           # 世界模型权重 (4.01 GB)
│           └── video_vqvae.pth           # VQVAE 权重 (0.92 GB)
│
├── 📂 配置文件
│   └── configs/
│       ├── train_consistency_mini.py     # 训练配置（mini 数据集）
│       └── ...
│
├── 📂 训练脚本
│   └── scripts/
│       └── dlc_train.sh                  # DLC 多卡训练（一键训练）
│
├── 📂 数据与索引
│   ├── indices/                          # 训练/验证索引
│   │   ├── consistency_train.jsonl
│   │   └── consistency_val.jsonl
│   └── work_dirs/                        # 训练输出
│       └── critic_full/
│           ├── checkpoints/
│           │   └── best.pth              # 最佳模型
│           ├── logs/
│           └── metrics/
│
└── 📂 文档
    └── docs/
        ├── framework_overview.md         # 本文件
        ├── consistency_critic_model.md   # Critic 模型设计
        └── critic_model.md               # 模型架构说明
```

---

## 🎯 核心组件

### 1. Consistency Critic 模型

**文件**: `train.py`

**功能**: 多维度动作一致性评分器

**架构**:
```python
class MultiDimensionalConsistencyCritic(nn.Module):
    """多维度一致性评分器"""
    
    def __init__(self):
        # 视觉编码器 (ResNet-50)
        self.vision_encoder = resnet50(pretrained=True)
        
        # 动作编码器 (MLP)
        self.action_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # 融合层
        self.fusion = nn.TransformerEncoder(...)
        
        # 多维度输出头
        self.consistency_head = nn.Linear(512, 2)   # consistency
        self.validity_head = nn.Linear(512, 2)       # validity
        self.speed_head = nn.Linear(512, 2)          # speed consistency
        self.steering_head = nn.Linear(512, 2)       # steering consistency
        self.progress_head = nn.Linear(512, 2)       # progress consistency
        self.temporal_head = nn.Linear(512, 1)       # temporal coherence
    
    def forward(self, images, actions):
        # 提取视觉特征
        vision_features = self.vision_encoder(images)
        
        # 提取动作特征
        action_features = self.action_encoder(actions)
        
        # 融合特征
        fused = self.fusion(vision_features, action_features)
        
        # 多维度输出
        return {
            'consistency': self.consistency_head(fused),
            'validity': self.validity_head(fused),
            'speed': self.speed_head(fused),
            'steering': self.steering_head(fused),
            'progress': self.progress_head(fused),
            'temporal': self.temporal_head(fused),
        }
```

**训练策略**:
- Multi-task learning（多任务学习）
- Weighted loss（加权损失）
- 动态权重调整

---

### 2. DrivingWorld 集成

**路径**: `DrivingWorld/`

**功能**: 使用世界模型生成训练数据

**权重**:
- `world_model.pth` (4.01 GB) - 世界模型
- `video_vqvae.pth` (0.92 GB) - 视频 VQVAE

**使用方式**:
```python
from generation.drivingworld_wrapper import DrivingWorldWrapper

world_model = DrivingWorldWrapper(
    config_path="DrivingWorld/configs/default_config.yaml",
    checkpoint_path="DrivingWorld/pretrained_models/world_model.pth",
    vqvae_path="DrivingWorld/pretrained_models/video_vqvae.pth",
    device="cuda"
)

# 生成未来帧
result = world_model.generate(
    history_images=history_frames,    # [4, 3, 256, 256]
    ego_state=ego_state,              # [speed, acceleration, steering]
    candidate_actions=trajectory      # [8, 6] 未来轨迹
)

generated_images = result['generated_frames']  # [8, 3, 256, 256]
```

---

### 3. 训练数据生成

**文件**: `generate_critic_training_data.py`

**策略**: 正负样本对生成

```python
# 正样本：使用真实轨迹
positive_sample = world_model.generate(
    history_images=history,
    candidate_actions=real_trajectory  # 真实轨迹
)
# 标签：consistency=1, validity=1

# 负样本：使用扰动轨迹
noise_level = random.uniform(0.1, 0.5)
noisy_trajectory = real_trajectory + noise
negative_sample = world_model.generate(
    history_images=history,
    candidate_actions=noisy_trajectory  # 扰动轨迹
)
# 标签：consistency=0, validity=0
```

**数据格式**:
```python
{
    'history_images': [...],        # 历史图像路径
    'future_images': [...],         # 真实未来图像
    'generated_images': [...],      # 生成图像
    'ego_state': [...],             # 自车状态
    'trajectory': [...],            # 规划轨迹
    'noise_level': 0.0,             # 噪声级别（正样本=0）
    'scene_id': 'scene_001',
    'sample_id': 0,
}
```

---

### 4. 标签计算

**文件**: `compute_training_labels.py`

**6 维度标签**:

| 维度 | 计算方法 | 阈值 |
|------|---------|------|
| consistency_label | FID < 50 | 50 |
| speed_consistency | 轨迹速度 vs 图像变化 | - |
| steering_consistency | 转向角 vs 水平移动 | - |
| progress_consistency | 前进运动 vs 图像变化 | - |
| temporal_coherence | 帧间变化平滑度 | - |
| validity_label | 综合判断 | 多个条件 |

**计算示例**:
```python
def compute_speed_consistency(generated, ground_truth, trajectory):
    # 从轨迹提取速度
    traj_speed = torch.sqrt(dx**2 + dy**2).mean()
    
    # 从图像变化估计速度
    img_speed = abs(generated[1:] - generated[:-1]).mean()
    
    # 一致性判断
    if traj_speed < 0.5 and img_speed < 0.1:
        return 1  # 低速 + 小变化 = 一致
    elif traj_speed > 1.0 and img_speed > 0.2:
        return 1  # 高速 + 大变化 = 一致
    else:
        return 0  # 不一致
```

---

### 5. DLC 多卡训练

**文件**: `scripts/dlc_train.sh`

**功能**: 一键训练（自动数据准备 + 多卡训练）

**使用**:
```bash
# 完整训练
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh

# 冒烟测试
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh --smoke-test

# 自定义参数
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh \
  --epochs=100 \
  --batch-size=32 \
  --work-dir=work_dirs/my_model
```

**特性**:
- ✅ 自动检测 GPU 数量
- ✅ 自动数据准备（如果索引不存在）
- ✅ 多节点多卡支持
- ✅ NCCL 优化（1800 秒超时）
- ✅ 文件描述符优化（65536）

---

## 🚀 快速开始

### 一行命令训练

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh
```

**自动流程**:
1. 检查训练数据
2. 如果不存在，自动生成（DrivingWorld）
3. 计算 6 维度标签
4. 构建训练索引
5. 启动多卡训练（50 epochs）
6. 保存最佳模型

---

### 手动分步执行

#### Step 1: 生成训练数据

```bash
python generate_critic_training_data.py \
  --data-root /mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set \
  --output-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data \
  --num-scenes 500 \
  --samples-per-scene 5 \
  --device cuda:0
```

**输出**: `critic_training_data/*.pt` (约 2500 个样本)

---

#### Step 2: 计算标签

```bash
python compute_training_labels.py \
  --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data \
  --output-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
  --device cuda:0
```

**输出**: 标注后的样本（6 维度标签）

---

#### Step 3: 构建索引

```bash
python build_critic_index.py \
  --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
  --output-dir indices \
  --train-ratio 0.8 \
  --balance-classes
```

**输出**:
- `indices/consistency_train.jsonl`
- `indices/consistency_val.jsonl`

---

#### Step 4: 训练模型

```bash
bash scripts/dlc_train.sh \
  --config configs/train_consistency_mini.py \
  --epochs 50 \
  --batch-size 16 \
  --work-dir work_dirs/critic_full
```

**输出**: `work_dirs/critic_full/checkpoints/best.pth`

---

#### Step 5: 评估模型

```bash
python eval_critic.py \
  --checkpoint work_dirs/critic_full/checkpoints/best.pth \
  --val-index indices/consistency_val.jsonl \
  --eval-ranking \
  --output-dir eval_results/critic_full
```

**输出**: 评估报告（Ranking Metrics + 相关性分析）

---

## 📊 训练配置

### 默认参数

```python
# configs/train_consistency_mini.py

data = dict(
    train_index="indices/consistency_train.jsonl",
    val_index="indices/consistency_val.jsonl",
    batch_size=16,
    num_workers=4,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=1e-2,
)

scheduler = dict(
    type="CosineAnnealingLR",
    T_max=50,
    eta_min=1e-6,
)

# 损失权重
loss_weights = dict(
    lambda_consistency=1.0,
    lambda_validity=0.5,
    lambda_speed_consistency=0.3,
    lambda_steering_consistency=0.3,
    lambda_progress_consistency=0.2,
    lambda_temporal_coherence=0.2,
)
```

---

### 多卡训练参数

| 配置 | 单卡 | 8 卡 | 16 卡 |
|------|------|------|-------|
| batch_size | 8 | 16 | 32 |
| epochs | 50 | 50 | 50 |
| num_workers | 4 | 4 | 4 |
| 学习率 | 1e-4 | 8e-4 | 16e-4 |
| 预计时间 | ~24h | ~3-4h | ~2h |

---

## 📈 评估指标

### Layer 1: 生成质量

| 指标 | 说明 | 范围 | 越低越好 |
|------|------|------|---------|
| FID | Fréchet Inception Distance | [0, ∞) | ✅ |
| FVD | Fréchet Video Distance | [0, ∞) | ✅ |
| LPIPS | 感知相似度 | [0, 1] | ✅ |

---

### Layer 2: Action 一致性

| 维度 | 说明 | 范围 | 越高越好 |
|------|------|------|---------|
| consistency | 整体一致性 | {0, 1} | ✅ |
| speed_consistency | 速度一致性 | {0, 1} | ✅ |
| steering_consistency | 转向一致性 | {0, 1} | ✅ |
| progress_consistency | 进度一致性 | {0, 1} | ✅ |
| temporal_coherence | 时序连贯性 | [0, 1] | ✅ |
| validity | 综合有效性 | {0, 1} | ✅ |

---

### Layer 3: 驾驶合理性

| 指标 | 说明 | 范围 | 越高越好 |
|------|------|------|---------|
| NDCG@3 | 归一化折损累积增益@3 | [0, 1] | ✅ |
| NDCG@5 | 归一化折损累积增益@5 | [0, 1] | ✅ |
| MRR | 平均倒数排名 | [0, 1] | ✅ |
| Top-1 Hit Rate | 首命中率 | [0, 1] | ✅ |

---

### 最终验证: 相关性分析

| 指标 | 说明 | 目标 |
|------|------|------|
| Spearman ρ | Critic vs nuPlan 闭环 | > FID/FVD 的相关性 |
| Kendall τ | 排名相关性 | > FID/FVD 的相关性 |

---

## 🎓 论文贡献

### 核心贡献

1. **多维度 Action-Conditioned Consistency Critic**
   - 首次提出 6 维度评分框架
   - 使用 World Model 自动生成训练数据
   - Multi-task learning 训练

2. **三层评估系统**
   - Layer 1: 生成质量（FID/FVD/LPIPS）
   - Layer 2: 动作一致性（6 维度）
   - Layer 3: 驾驶合理性（Ranking）

3. **实证验证**
   - 证明 Critic 与 nuPlan 闭环性能的相关性 > FID/FVD
   - 提供完整的训练和评估流程

### 创新点

- ✅ 首次使用 World Model 生成训练数据
- ✅ 首次提出多维度 Action-Conditioned 评分
- ✅ 首次证明 visual-action metric 与规划性能的相关性

---

## 💡 使用场景

### 场景 1: 训练新模型

```bash
# 一行命令
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh
```

---

### 场景 2: 评估已有模型

```bash
python eval_critic.py \
  --checkpoint work_dirs/critic_full/checkpoints/best.pth \
  --val-index indices/consistency_val.jsonl \
  --eval-ranking
```

---

### 场景 3: 生成新数据

```bash
python generate_critic_training_data.py \
  --num-scenes 1000 \
  --samples-per-scene 5
```

---

### 场景 4: 三层评估

```bash
python evaluate_three_layers.py \
  --generated-dir /path/to/generated \
  --groundtruth-dir /path/to/groundtruth \
  --val-index indices/consistency_val.jsonl
```

---

## ⚠️ 注意事项

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 显存: ≥ 10 GB（单卡）
- 磁盘: ≥ 100 GB

---

### 数据路径

```bash
# nuPlan 数据
/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set

# 训练数据
/mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data

# 标注数据
/mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled

# 训练索引
/mnt/cpfs/prediction/lipeinan/nuplan/indices/
```

---

### 常见问题

**Q: 训练 OOM 怎么办？**

```bash
# 减小 batch size
bash scripts/dlc_train.sh --batch-size=8
```

---

**Q: 如何查看训练进度？**

```bash
tail -f work_dirs/critic_full/logs/training.log
```

---

**Q: 训练中断了怎么办？**

```bash
# 重新运行（数据已存在，会跳过生成）
bash scripts/dlc_train.sh
```

---

## 📚 相关文档

- [一键训练指南](../一键训练指南.md) - DLC 训练详细使用说明
- [三层评估使用指南](../三层评估使用指南.md) - 评估系统使用
- [DLC 多卡训练指南](../DLC多卡训练指南.md) - 多卡训练配置
- [Consistency Critic 模型](consistency_critic_model.md) - 模型设计文档
- [Critic 模型架构](critic_model.md) - 技术细节

---

## 🎊 总结

### 框架特点

- ✅ **完整自动化**: 从数据生成到模型训练全流程自动化
- ✅ **多维度评估**: 6 维度 Action-Conditioned 评分
- ✅ **高效训练**: DLC 多卡训练，3-4 小时完成
- ✅ **一键启动**: `bash scripts/dlc_train.sh`
- ✅ **实证验证**: 证明与闭环性能的相关性

### 核心文件

| 文件 | 功能 |
|------|------|
| `scripts/dlc_train.sh` | 一键训练 |
| `train.py` | 训练框架 |
| `eval_critic.py` | 评估脚本 |
| `generate_critic_training_data.py` | 数据生成 |
| `compute_training_labels.py` | 标签计算 |
| `build_critic_index.py` | 索引构建 |

### 快速命令

```bash
# 训练
cd /mnt/cpfs/prediction/lipeinan/nuplan && bash scripts/dlc_train.sh

# 评估
python eval_critic.py --checkpoint work_dirs/critic_full/checkpoints/best.pth

# 冒烟测试
bash scripts/dlc_train.sh --smoke-test
```

---

**🎉 框架说明完成！祝研究顺利！**
