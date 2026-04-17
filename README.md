# NuPlan Action-Conditioned Critic 训练框架

## 项目概述

本项目实现了一个 **Action-Conditioned Future Consequence Evaluator**（动作条件化未来结果评估器），用于评估自动驾驶场景中给定当前观测和动作候选，未来轨迹的合理性。

**核心功能**：
- 学习 nuPlan 驾驶数据分布下，图像观测与动作一致性的统计规律
- 提供多维度评估能力（一致性、速度、转向、前进、时间连贯性、驾驶合理性）
- 支持 Ranking 评估（NDCG、MRR、Top-1 Hit Rate）
- 完整的训练 pipeline，支持本地和分布式训练

---

## Pipeline 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                              │
├─────────────────────────────────────────────────────────────────┤
│  nuPlan DB 文件  ──┐                                            │
│                    ├──► 索引构建 ──► train/val JSONL 索引文件    │
│  相机图像目录  ────┘                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        模型训练阶段                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入:                                                           │
│  ├─ 历史图像序列 (CAM_F0, 4帧)                                  │
│  ├─ 当前 Ego 状态 (速度、航向、加速度、角速度)                   │
│  └─ 候选未来轨迹 (8步, 每步0.5s)                                │
│                                                                  │
│  模型:                                                           │
│  ├─ Image Encoder (CNN + Projection)                            │
│  ├─ Trajectory Encoder (MLP)                                    │
│  ├─ Ego State Encoder (MLP)                                     │
│  └─ 多维度评估头 (6个head)                                      │
│                                                                  │
│  训练:                                                           │
│  ├─ 正样本: 真实轨迹 (label=1)                                  │
│  ├─ 负样本: 随机错配轨迹 (label=0)                              │
│  └─ Loss: 多维度加权 BCE                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        评估阶段                                  │
├─────────────────────────────────────────────────────────────────┤
│  ├─ 分类准确率 (Accuracy)                                       │
│  ├─ Ranking 指标 (NDCG@3/5, MRR, Top-1)                         │
│  └─ 多维度一致性分数                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
nuplan/
├── README.md                      # 项目说明文档
├── train.py                       # 训练主程序（支持本地/分布式）
├── eval_critic.py                 # 模型评估脚本
├── closed_loop_evaluation.py      # 闭环评估（待完善）
│
├── configs/                       # 配置文件目录
│   ├── train_critic_mini.py       # 基础 critic 训练配置
│   └── train_consistency_mini.py  # 多维度一致性 critic 配置
│
├── tools/                         # 工具脚本
│   ├── build_critic_index.py      # 构建基础 critic 索引
│   └── build_consistency_index.py # 构建一致性 critic 索引
│
├── indices/                       # 数据索引（JSONL格式）
│   ├── critic_train.jsonl         # 训练集索引
│   ├── critic_val.jsonl           # 验证集索引
│   ├── consistency_train.jsonl    # 一致性训练集索引
│   └── consistency_val.jsonl      # 一致性验证集索引
│
├── generation/                    # 未来图像生成模块
│   ├── drivingworld_wrapper.py    # DrivingWorld 集成
│   └── drivewm_wrapper.py         # DriveWM 集成
│
├── evaluation/                    # 评估指标计算
│   ├── fid_calculator.py          # FID 计算
│   ├── fvd_calculator.py          # FVD 计算
│   └── lpips_calculator.py        # LPIPS 计算
│
├── scripts/                       # 运行脚本
│   └── dlc_train.sh               # DLC 平台分布式训练入口
│
├── work_dirs/                     # 实验输出目录
│   └── {experiment_name}/
│       ├── checkpoints/           # 模型检查点
│       ├── config_snapshot.json   # 配置快照
│       └── eval_results.json      # 评估结果
│
└── DrivingWorld/                  # World Model 子模块（git submodule）
```

---

## 数据流与工作流程

### 1. 数据准备

#### 1.1 数据依赖

项目需要以下数据：

**nuPlan 数据库文件**：
```bash
/mnt/datasets/e2e-datasets/20260227/e2e-datasets/dataset_pkgs/nuplan-v1.1/splits/mini
```

**相机图像目录**：
```bash
/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set/
├── nuplan-v1.1_mini_camera_0/
└── nuplan-v1.1_mini_camera_1/
```

#### 1.2 为什么需要 DB 文件

图像文件名是 token（如 `14fffca1394c537d.jpg`），无法直接从文件名推断时序关系。必须通过 DB 中的三张表对齐：
- `image` 表：图像元数据
- `camera` 表：相机通道信息
- `ego_pose` 表：车辆位姿

### 2. 索引构建

索引构建是将原始数据转换为训练可用的样本格式：

```bash
# 构建基础 critic 索引
python tools/build_critic_index.py

# 构建多维度一致性 critic 索引
python tools/build_consistency_index.py

# 调试模式（仅构建少量样本）
python tools/build_critic_index.py --max-scenes 2 --max-samples-per-scene 20
```

**输出格式**（JSONL，每行一个样本）：
```json
{
  "sample_id": "2021.05.25.14.16.10_veh-35_01690_02183__1621968321487272__pos",
  "scene_name": "2021.05.25.14.16.10_veh-35_01690_02183",
  "timestamp_us": 1621968321487272,
  "history_images": [
    "nuplan-v1.1_mini_camera_0/.../CAM_F0/xxxx.jpg",
    "nuplan-v1.1_mini_camera_0/.../CAM_F0/yyyy.jpg"
  ],
  "ego_state": [vx, vy, yaw, acceleration_x, angular_rate_z],
  "candidate_traj": [[dx1, dy1, dyaw1], [dx2, dy2, dyaw2], ...],
  "label": 1
}
```

**正负样本策略**：
- **正样本**：历史图像 + 真实未来轨迹 → label=1
- **负样本**：历史图像 + 随机错配轨迹 → label=0

### 3. 模型训练

#### 3.1 本地训练

```bash
# 标准训练
python train.py --config configs/train_critic_mini.py

# 自定义工作目录
python train.py --config configs/train_critic_mini.py --work-dir ./work_dirs/exp1

# 调试模式（快速验证）
python train.py \
  --config configs/train_critic_mini.py \
  --work-dir ./work_dirs/smoke_test \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-steps 3 \
  --max-val-steps 2
```

#### 3.2 DLC 分布式训练

```bash
# 默认配置
bash scripts/dlc_train.sh

# 自定义参数
bash scripts/dlc_train.sh --batch-size 16 --epochs 20 --work-dir ./work_dirs/critic_bs16
```

**分布式训练特性**：
- 自动读取 DLC 注入的环境变量（RANK、WORLD_SIZE、LOCAL_RANK）
- 使用 `torch.distributed.run` 启动多进程
- 支持多机多卡训练（timeout=30分钟）

#### 3.3 训练输出

每次训练会在 `work_dirs/{experiment_name}/` 下生成：
```
work_dirs/critic_mini_v1/
├── checkpoints/
│   ├── latest.pth          # 最新检查点
│   └── best.pth            # 验证集最优检查点
├── config_snapshot.json    # 配置快照（用于复现）
└── eval_val_results.json   # 验证集评估结果
```

### 4. 模型评估

```bash
# 基础评估（准确率）
python eval_critic.py --checkpoint work_dirs/critic_mini_v1/checkpoints/best.pth

# Ranking 评估
python eval_critic.py \
  --checkpoint work_dirs/consistency_mini_v1/checkpoints/best.pth \
  --eval-ranking
```

**评估指标**：
- **分类准确率**：判断正负样本的准确性
- **NDCG@3/5**：归一化折扣累积增益
- **MRR**：平均倒数排名
- **Top-1 Hit Rate**：最佳候选命中率

### 5. 未来图像生成（扩展功能）

集成 World Model 生成未来图像：

```bash
# 使用 DrivingWorld 生成
python generate_futures_drivingworld.py \
  --checkpoint /path/to/drivingworld_ckpt.pth \
  --input-data /path/to/input.jsonl \
  --output-dir ./generated_data
```

**生成数据用途**：
- 计算 FID/FVD 指标（生成质量评估）
- 构建生成图像数据集
- 用于后续一致性评估

### 6. 闭环评估（开发中）

```bash
# 闭环性能评估
python closed_loop_evaluation.py \
  --critic-checkpoint work_dirs/critic_mini_v1/checkpoints/best.pth \
  --scenarios /path/to/scenarios
```

**评估目标**：
- Critic score 与 nuPlan 闭环性能的相关性
- 对比传统 FID/FVD 指标的相关性

---

## 模型架构

### 基础 Critic 模型

```
历史图像 (4帧) ──► SimpleImageEncoder ──┐
                                         ├──► Concat ──► Head ──► Score
候选未来轨迹 ──► TrajectoryEncoder ─────┤
                                         │
Ego 状态 ──────► EgoEncoder ────────────┘
```

**组件说明**：
- **SimpleImageEncoder**：对历史图像逐帧编码（CNN），时间维度平均池化
- **TrajectoryEncoder**：未来轨迹展平后通过 MLP 编码
- **EgoEncoder**：Ego 状态通过 MLP 编码
- **Head**：拼接三类特征，输出二分类 logit

### 多维度 Consistency Critic 模型

```
                    ┌─► Consistency Head ───► L_consistency
                    ├─► Speed Head ─────────► L_speed
历史图像 ──┐        ├─► Steering Head ──────► L_steering
           ├─► Shared Backend ──┤
未来图像 ──┘        ├─► Progress Head ──────► L_progress
                    ├─► Temporal Head ──────► L_temporal
                    └─► Validity Head ──────► L_validity

候选轨迹 ──► Traj Encoder ──┘
Ego 状态 ──► Ego Encoder ───┘
```

**多维度评估头**：
1. **consistency_head**：总体一致性
2. **speed_consistency_head**：速度一致性
3. **steering_consistency_head**：转向一致性
4. **progress_consistency_head**：前进一致性
5. **temporal_coherence_head**：时间连贯性
6. **validity_head**：驾驶合理性

**损失函数**：
```
L = λ_c * L_consistency + λ_v * L_validity
  + λ_speed * L_speed + λ_steering * L_steering
  + λ_progress * L_progress + λ_temporal * L_temporal
```

---

## 配置参数说明

### 核心配置项（configs/train_critic_mini.py）

```python
cfg = dict(
    # 实验配置
    experiment_name="nuplan_critic_mini_v1",  # 实验名称
    seed=42,                                   # 随机种子
    work_dir="./work_dirs/critic_mini_v1",    # 输出目录
    
    # 数据路径
    train_index="indices/critic_train.jsonl",  # 训练索引
    val_index="indices/critic_val.jsonl",      # 验证索引
    image_root="/path/to/nuplan_data/mini_set", # 图像根目录
    mini_db_root="/path/to/mini/db",           # DB 文件目录
    camera_roots=[...],                         # 相机数据目录列表
    
    # 数据参数
    camera_channel="CAM_F0",                   # 使用的相机通道
    history_num_frames=4,                      # 历史帧数
    candidate_traj_steps=8,                    # 未来轨迹步数
    future_step_time_s=0.5,                    # 轨迹采样间隔（秒）
    image_size=224,                            # 图像尺寸
    
    # 训练参数
    epochs=10,                                 # 训练轮数
    batch_size=8,                              # 每卡 batch size
    num_workers=4,                             # DataLoader worker 数
    log_interval=20,                           # 日志打印间隔
    val_interval=1,                            # 验证间隔（epoch）
    save_interval=1,                           # 保存间隔（epoch）
    
    # 优化器
    optimizer=dict(
        lr=1e-4,                               # 学习率
        weight_decay=1e-2,                     # 权重衰减
    ),
    
    # 模型参数
    model=dict(
        image_channels=3,
        image_feature_dim=256,
        action_feature_dim=128,
        hidden_dim=256,
        dropout=0.1,
    ),
    
    # 数据集参数
    dataset=dict(
        normalize_ego_state=True,              # 是否归一化 ego 状态
        normalize_candidate_traj=True,         # 是否归一化轨迹
        image_mean=[0.485, 0.456, 0.406],     # 图像归一化均值
        image_std=[0.229, 0.224, 0.224],      # 图像归一化标准差
    ),
)
```

---

## 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0（GPU 训练）

### 5 分钟上手

```bash
# 1. 构建索引（调试模式）
python tools/build_critic_index.py --max-scenes 2 --max-samples-per-scene 20

# 2. 运行冒烟测试
python train.py \
  --config configs/train_critic_mini.py \
  --work-dir ./work_dirs/smoke_test \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-steps 3 \
  --max-val-steps 2

# 3. 查看结果
ls -lh work_dirs/smoke_test/checkpoints/
```

---

## 扩展与定制

### 更换 Backbone

修改 `train.py` 中的模型定义，替换为更强的图像编码器（如 ResNet-50、ViT 等）。

### 多相机输入

在配置文件中修改：
```python
camera_channel = ["CAM_F0", "CAM_L0", "CAM_R0"]  # 使用多个相机
history_num_frames = 4                            # 每个相机的历史帧数
```

### 自定义负样本策略

修改 `tools/build_critic_index.py` 中的负样本生成逻辑：
- 难负样本：选择相似的错误轨迹
- 课程学习：逐步增加负样本难度

### 集成其他 World Model

在 `generation/` 目录下添加新的 wrapper：
```python
# generation/my_world_model_wrapper.py
class MyWorldModelWrapper:
    def generate(self, ...):
        # 实现生成逻辑
        pass
```

---

## 常见问题

### Q: 为什么必须使用 DB 文件？

A: 图像文件名是随机 token，无法推断时序。DB 文件提供了图像、相机、位姿的关联关系。

### Q: 如何加快索引构建速度？

A: 使用 `--max-scenes` 和 `--max-samples-per-scene` 参数限制样本数量。

### Q: 训练时显存不足怎么办？

A: 减小 `batch_size` 或 `history_num_frames`，或使用梯度累积。

### Q: 如何恢复训练？

A: 修改配置文件中的 `work_dir`，在 `train.py` 中添加 checkpoint 加载逻辑。

### Q: 评估指标很低怎么办？

A: 
- 检查数据质量（索引文件是否正确）
- 增加训练轮数
- 调整学习率
- 使用更强的 backbone

---

## 技术栈

- **深度学习框架**：PyTorch
- **分布式训练**：torch.distributed (NCCL)
- **数据处理**：PIL, NumPy
- **配置管理**：Python 字典配置
- **实验管理**：自定义 work_dirs 结构

---

## 开发计划

### Phase 1: 基础 Critic（已完成 ✅）
- [x] 单相机输入（CAM_F0）
- [x] 多维度评估头
- [x] Ranking 评估能力
- [x] 本地 + 分布式训练支持

### Phase 2: World Model 集成（开发中）
- [ ] DrivingWorld 完整集成
- [ ] 图像生成 pipeline
- [ ] FID/FVD 计算

### Phase 3: 闭环验证（规划中）
- [ ] nuPlan closed-loop simulator 集成
- [ ] Critic-guided planning
- [ ] 相关性分析实验

### Phase 4: 生产部署（规划中）
- [ ] 性能优化
- [ ] 多相机多模态
- [ ] 在线评估接口

---

## 许可与引用

本项目基于 nuPlan 数据集开发，请遵守 nuPlan 的数据使用协议。

---

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
