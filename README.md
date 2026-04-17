# NuPlan Critic 训练框架 - 三层评估体系

## 0. 项目定位

本项目实现的是一个 **Action-Conditioned Future Consequence Evaluator**（动作条件化未来结果评估器）。

**核心目标**：学习在 nuPlan 驾驶数据分布下，给定当前图像和 ego action，合理未来图像应该长成什么样的统计规律。

**三层评估框架**：
- **Layer 1**: 生成质量评估（生成图像 vs 真实图像的相似度）
- **Layer 2**: Action一致性评估（生成未来是否符合给定action）
- **Layer 3**: 驾驶合理性评估（未来是否驾驶合理、安全）

**最终目标**：证明所提 visual-action metric 与 nuPlan 闭环性能的相关性高于传统 FID/FVD。

---

## 1. 项目目标

本目录实现的是一个 `NuPlan Action-Conditioned Critic` 的第一版最小可训练框架。

当前版本目标不是直接生成未来，而是学习一个打分器：

- 输入：
  - 历史前视图像
  - 当前 ego 状态
  - 一段候选未来轨迹
- 输出：
  - 一个 `match / mismatch` 分数

这个分数可以理解为：

- 当前图像观察和候选动作是否一致
- 当前状态是否真的会导向该候选未来轨迹

当前版本属于第一阶段：

- 已支持：`history image + ego state + candidate trajectory -> score`
- 未支持：`candidate future image` 作为额外输入

后续可以在这个框架基础上扩展到完整的图像版 critic。

---

## 2. 当前目录结构

```text
/mnt/cpfs/prediction/lipeinan/nuplan/
  README.md
  use.md
  dlc_train.sh
  train.py
  configs/
    train_critic_mini.py
  tools/
    build_critic_index.py
  indices/
    critic_train.jsonl
    critic_val.jsonl
    critic_index_summary.json
  work_dirs/
```

各文件作用如下：

- `use.md`
  - 原始任务描述
- `README.md`
  - 当前这份说明文档
- `dlc_train.sh`
  - DLC 平台训练入口脚本
- `train.py`
  - 第一版训练主程序
- `configs/train_critic_mini.py`
  - 第一版训练配置
- `tools/build_critic_index.py`
  - 从 `nuPlan mini db + mini_set 相机图片` 生成训练索引
- `indices/critic_train.jsonl`
  - 训练集索引
- `indices/critic_val.jsonl`
  - 验证集索引
- `indices/critic_index_summary.json`
  - 索引构建摘要

---

## 3. 数据依赖

### 3.1 图片目录

当前使用的是你已经下载并解压的 `mini_set` 相机图片：

```text
/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set
```

目前已经使用到：

- `nuplan-v1.1_mini_camera_0`
- `nuplan-v1.1_mini_camera_1`

单个场景目录结构类似：

```text
nuplan-v1.1_mini_camera_0/
  2021.05.25.14.16.10_veh-35_01690_02183/
    CAM_F0/
    CAM_B0/
    CAM_L0/
    CAM_L1/
    CAM_L2/
    CAM_R0/
    CAM_R1/
    CAM_R2/
```

当前训练版本只使用：

- `CAM_F0`

### 3.2 mini db 目录

当前索引构建使用的是：

```text
/mnt/datasets/e2e-datasets/20260227/e2e-datasets/dataset_pkgs/nuplan-v1.1/splits/mini
```

这些 `db` 文件提供了：

- 图像文件名 `filename_jpg`
- 图像时间戳 `timestamp`
- ego pose
- ego 速度
- 相机通道信息

### 3.3 为什么必须用 db

图片目录中的文件名是 token，例如：

```text
14fffca1394c537d.jpg
```

它不是简单的帧号，因此不能直接靠文件名恢复时序。  
必须通过 `db` 中的：

- `image`
- `camera`
- `ego_pose`

三张表做对齐。

---

## 4. 第一版任务定义

### 4.1 输入

- 历史图像：
  - 默认取 `CAM_F0`
  - 默认取连续 `4` 帧
- 当前 ego 状态：
  - 当前速度
  - 当前航向
  - 当前加速度
  - 当前角速度
- 候选未来轨迹：
  - 默认 `8` 步
  - 每步是 `(dx, dy, dyaw)`
  - 默认采样间隔 `0.5s`

### 4.2 输出

- 一个二分类 logit
- 训练时对应 `label in {0, 1}`

### 4.3 正负样本

正样本：

- 当前历史图像
- 当前 ego 状态
- 当前时刻真实 future trajectory
- `label = 1`

负样本：

- 当前历史图像不变
- ego 状态不变
- 从同一 split 中随机取另一条候选 future trajectory
- `label = 0`

这是第一版最简单的 critic 构造方法，便于先跑通训练链路。

---

## 5. 当前模型结构 - 多维度评估

`train.py` 中包含两个版本的模型：

### 5.1 基础 Critic 模型（CriticModel）
- `SimpleImageEncoder`
  - 对历史图像逐帧编码
  - 再做时间维平均
- `TrajectoryEncoder`
  - 对未来轨迹展平后做 MLP 编码
- `ego encoder`
  - 对 ego 状态做 MLP 编码
- `head`
  - 拼接三类特征
  - 输出一个二分类分数

### 5.2 多维度 Consistency Critic 模型（ConsistencyCriticModel）

**共享 Backbone**：
- History/Future Image Encoder（共享CNN，独立投影）
- Trajectory Encoder
- Ego Encoder

**多维度评估头**：
- `consistency_head`: overall consistency（总体一致性）
- `speed_consistency_head`: speed consistency（速度一致性）
- `steering_consistency_head`: steering consistency（转向一致性）
- `progress_consistency_head`: progress consistency（前进一致性）
- `temporal_coherence_head`: temporal coherence（时间连贯性）
- `validity_head`: driving validity（驾驶合理性）

**训练 Loss**：
```
L = λ_c * L_consistency + λ_v * L_validity
  + λ_speed * L_speed + λ_steering * L_steering
  + λ_progress * L_progress + λ_temporal * L_temporal
```

这是多维度评估版本，后续可以替换成更强 backbone 或 ranking loss。

---

## 5.1 Ranking 评估能力

除了基础的分类准确率，模型还需要具备良好的排序能力：

**评估指标**：
- `NDCG@3`, `NDCG@5`: 归一化折扣累积增益
- `MRR`: 平均倒数排名
- `Top-1 Hit Rate`: 最佳候选命中率

**评估方法**：
```bash
python eval_critic.py --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth --eval-ranking
```

---

## 6. 配置文件说明

配置文件位于：

```text
/mnt/cpfs/prediction/lipeinan/nuplan/configs/train_critic_mini.py
```

其中重要字段包括：

- `train_index`
  - 训练索引路径
- `val_index`
  - 验证索引路径
- `image_root`
  - 图片根目录
- `mini_db_root`
  - mini split db 路径
- `camera_roots`
  - 当前已解压的相机包目录
- `camera_channel`
  - 当前使用的相机，默认 `CAM_F0`
- `history_num_frames`
  - 历史帧数，默认 `4`
- `candidate_traj_steps`
  - future 轨迹步数，默认 `8`
- `future_step_time_s`
  - future 轨迹采样时间间隔，默认 `0.5`
- `epochs`
  - 默认训练 epoch
- `batch_size`
  - 每卡 batch size
- `num_workers`
  - dataloader worker 数

---

## 7. 索引文件格式

训练和验证集最终读取的是 `jsonl` 文件。

每条样本最少包含：

```json
{
  "sample_id": "2021.05.25.14.16.10_veh-35_01690_02183__1621968321487272__pos",
  "scene_name": "2021.05.25.14.16.10_veh-35_01690_02183",
  "timestamp_us": 1621968321487272,
  "history_images": [
    "nuplan-v1.1_mini_camera_0/2021.05.25.14.16.10_veh-35_01690_02183/CAM_F0/xxxx.jpg",
    "nuplan-v1.1_mini_camera_0/2021.05.25.14.16.10_veh-35_01690_02183/CAM_F0/yyyy.jpg"
  ],
  "ego_state": [vx, vy, yaw, acceleration_x, angular_rate_z],
  "candidate_traj": [
    [dx1, dy1, dyaw1],
    [dx2, dy2, dyaw2]
  ],
  "label": 1
}
```

负样本格式相同，只是：

- `candidate_traj` 换成错配轨迹
- `label = 0`

---

## 8. 如何重新生成索引

当前索引由这个脚本生成：

```text
/mnt/cpfs/prediction/lipeinan/nuplan/tools/build_critic_index.py
```

### 8.1 默认生成

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan
python tools/build_critic_index.py
```

生成结果会写到：

```text
/mnt/cpfs/prediction/lipeinan/nuplan/indices/
```

包括：

- `critic_train.jsonl`
- `critic_val.jsonl`
- `critic_index_summary.json`

### 8.2 调试模式

如果只想抽少量场景快速验证：

```bash
python tools/build_critic_index.py --max-scenes 2 --max-samples-per-scene 20
```

### 8.3 常见可调参数

- `--camera-channel`
  - 默认 `CAM_F0`
- `--history-num-frames`
  - 默认 `4`
- `--future-steps`
  - 默认 `8`
- `--future-step-time-s`
  - 默认 `0.5`
- `--sample-stride`
  - 默认 `5`
- `--val-ratio`
  - 默认 `0.2`

---

## 9. 本地训练方法

### 9.1 正常训练

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan
python train.py --config configs/train_critic_mini.py
```

### 9.2 覆盖 work_dir

```bash
python train.py \
  --config configs/train_critic_mini.py \
  --work-dir ./work_dirs/critic_exp1
```

### 9.3 调试冒烟测试

```bash
python train.py \
  --config configs/train_critic_mini.py \
  --work-dir ./work_dirs/smoke_test \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-steps 3 \
  --max-val-steps 2
```

---

## 10. DLC 平台训练方法

当前 DLC 启动脚本：

```text
/mnt/cpfs/prediction/lipeinan/nuplan/dlc_train.sh
```

### 10.1 平台固定入口

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan
bash dlc_train.sh
```

### 10.2 覆盖部分训练参数

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan
bash dlc_train.sh --batch-size 16 --epochs 20 --work-dir ./work_dirs/critic_bs16
```

### 10.3 脚本做了什么

`dlc_train.sh` 会：

- 读取 DLC 注入的多机分布式环境变量
- 自动推断单机 GPU 数
- 使用 `torch.distributed.run` 启动训练
- 调用：

```text
train.py --config configs/train_critic_mini.py
```

---

## 11. 当前已验证通过的内容

当前已验证：

- `dlc_train.sh` shell 语法通过
- `train.py` Python 语法通过
- `build_critic_index.py` Python 语法通过
- lints 检查通过
- mini 场景索引可以正确生成
- 训练链路 smoke test 成功跑通

当前索引摘要见：

```text
/mnt/cpfs/prediction/lipeinan/nuplan/indices/critic_index_summary.json
```

---

## 12. 当前版本的局限

这仍然是第一版骨架，不是最终完整版。

当前限制：

- 只使用 `CAM_F0`
- 只使用历史图像，不使用 future image（基础版）
- 负样本仍较简单
- 模型 backbone 仍较轻量
- 没有专门的 evaluator / ranking 指标（基础版）
- **尚未与闭环性能验证**（这是最终目标）

---

## 13. 下一步建议 - 三层评估框架

### Phase 1: 强化当前 Critic（当前已完成）

✅ 多维度评估头（speed, steering, progress, temporal）
✅ Ranking 评估能力（NDCG, MRR, Top-1 Hit Rate）
✅ 细粒度负样本构造

### Phase 2: 集成 World Model（待实现）

- [ ] 集成 DrivingWorld 或其他 world model
- [ ] 实现图像生成 pipeline
- [ ] 添加 FID/FVD 计算（Layer 1: 生成质量）
- [ ] 构建生成图像数据集

### Phase 3: 逆动力学一致性（待实现）

- [ ] 训练/集成 inverse dynamics model
- [ ] 实现：生成图像 → 反推 action → 与条件比较
- [ ] 多维度 recoverability 指标（Layer 2: Action一致性）

### Phase 4: 闭环性能验证（核心目标）

- [ ] 集成 nuPlan closed-loop simulator
- [ ] 构建 critic-guided planning pipeline
- [ ] 设计相关性实验：
  - 计算 critic score vs closed-loop performance 的相关性
  - 计算 FID/FVD vs closed-loop performance 的相关性
  - **假设**：critic score 的相关性 > FID/FVD 的相关性
- [ ] 证明 visual-action metric 更能预测驾驶性能（Layer 3: 驾驶有用性）

### Phase 5: Model Rollout 适配

- [ ] 收集 DrivingWorld rollout 数据
- [ ] 两阶段训练：GT pretrain → model finetune
- [ ] 验证 domain gap 缩小

---

## 14. 一句话总结

当前这套工程已经具备：

- `mini_set 图片 + mini db -> 训练索引`
- `训练索引 -> 多维度 critic 训练`（consistency, speed, steering, progress, temporal, validity）
- `本地训练 + DLC 分布式训练入口`
- `Ranking 评估能力`（NDCG, MRR, Top-1 Hit Rate）

也就是说，**Phase 1 已经到达“可以正式开始训练”的状态**。

下一步核心目标：**证明所提 visual-action metric 与 nuPlan 闭环性能的相关性高于传统 FID/FVD**。

---

## 15. 核心贡献点

如果能证明：

```
Proposed Visual-Action Metric ⊨ nuPlan Closed-Loop Performance
```

比 FID/FVD 更能预测驾驶性能，这就是一个 **strong contribution**：

**标题示例**：
> "Beyond FID: Action-Consistent Evaluation for Driving World Models"

**论文叙事**：
```
Problem: FID/FVD 不能反映 driving usefulness
Method: Action-conditioned consequence evaluator
Key Finding: Our metric correlates with closed-loop performance (ρ=0.XX)
Impact: Better evaluation criterion for driving world models
```
