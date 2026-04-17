# NuPlan Critic 训练框架使用指南

## 📋 快速开始

### 1. 生成训练索引

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan

# 基础 Critic 索引（二分类）
python tools/build_critic_index.py

# 多维度 Consistency Critic 索引（推荐）
python tools/build_consistency_index.py
```

### 2. 训练模型

```bash
# 基础 Critic 训练
python train.py --config configs/train_critic_mini.py

# 多维度 Consistency Critic 训练（推荐）
python train.py --config configs/train_consistency_mini.py

# 调试模式（快速验证）
python train.py \
  --config configs/train_consistency_mini.py \
  --work-dir ./work_dirs/smoke_test \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-steps 3 \
  --max-val-steps 2
```

### 3. 评估模型

```bash
# 基础评估（Accuracy, F1, AUC）
python eval_critic.py \
  --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth

# 完整评估（包含 Ranking 能力）
python eval_critic.py \
  --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth \
  --eval-ranking

# 限制样本数（快速验证）
python eval_critic.py \
  --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth \
  --eval-ranking \
  --max-samples 100
```

---

## 🎯 三层评估框架

### Layer 1: 生成质量评估

**目标**: 评估生成图像与真实图像的相似度

**指标**（待实现）:
- FID (Fréchet Inception Distance)
- FVD (Fréchet Video Distance)
- LPIPS (Perceptual Similarity)

**当前状态**: ⚠️ 需要先集成 World Model

---

### Layer 2: Action一致性评估

**目标**: 评估生成未来是否符合给定 action

**多维度评估头**:
- `consistency`: 总体一致性
- `speed_consistency`: 速度一致性
- `steering_consistency`: 转向一致性
- `progress_consistency`: 前进一致性
- `temporal_coherence`: 时间连贯性

**Ranking 能力**:
- `NDCG@3`, `NDCG@5`: 归一化折扣累积增益
- `MRR`: 平均倒数排名
- `Top-1 Hit Rate`: 最佳候选命中率

**训练配置**:
```python
# configs/train_consistency_mini.py
lambda_consistency=1.0          # 总体一致性权重
lambda_speed_consistency=0.3    # 速度一致性权重
lambda_steering_consistency=0.3 # 转向一致性权重
lambda_progress_consistency=0.2 # 前进一致性权重
lambda_temporal_coherence=0.2   # 时间连贯性权重
```

---

### Layer 3: 驾驶合理性评估

**目标**: 评估未来是否驾驶合理、安全

**评估头**:
- `validity`: 驾驶合理性分数

**训练配置**:
```python
lambda_validity=0.5              # 驾驶合理性权重
validity_positive_weight=1.0     # 正样本权重
```

---

## 📊 训练输出说明

### 训练日志示例

```
[Epoch 1/30] loss=0.6234 c_acc=0.7123 v_acc=0.6845 speed_acc=0.7456 steering_acc=0.7234 progress_acc=0.6987 temporal_acc=0.7123 val_loss=0.5987 val_c_acc=0.7234 val_v_acc=0.6923
```

**指标含义**:
- `loss`: 总损失（多维度加权）
- `c_acc`: 总体一致性准确率
- `v_acc`: 驾驶合理性准确率
- `speed_acc`: 速度一致性准确率
- `steering_acc`: 转向一致性准确率
- `progress_acc`: 前进一致性准确率
- `temporal_acc`: 时间连贯性准确率

### 评估输出示例

```
============================================================
Consistency Critic 评估结果
============================================================
  总样本数: 5000
  [Consistency Head]
    正/负样本数: 1250 / 3750
    Accuracy:  0.7234
    Precision: 0.6987
    Recall:    0.7456
    F1 Score:  0.7214
    AUC:       0.7891
  
  [Ranking Metrics]
    场景数: 500
    NDCG@3:  0.7654
    NDCG@5:  0.7823
    MRR:     0.7234
    Top-1 Hit Rate: 0.6987
============================================================
```

---

## 🔧 高级配置

### 多维度权重调优

如果某个维度效果不好，可以调整权重：

```python
# configs/train_consistency_mini.py
# 增加 steering 一致性权重
lambda_steering_consistency=0.5  # 从 0.3 提升到 0.5

# 降低 temporal 权重
lambda_temporal_coherence=0.1    # 从 0.2 降低到 0.1
```

### Ranking 评估配置

```python
# configs/train_consistency_mini.py
ranking=dict(
    enabled=True,
    num_candidates_per_scene=5,       # 每个scene的候选数
    ranking_metrics=["ndcg@3", "ndcg@5", "mrr", "top1_hit_rate"],
)
```

**注意**: Ranking 评估需要索引中包含多个候选样本（同一 scene 的不同时刻或扰动）。

---

## 🚀 分布式训练

### DLC 平台训练

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan
bash dlc_train.sh

# 覆盖参数
bash dlc_train.sh --batch-size 16 --epochs 20 --work-dir ./work_dirs/critic_bs16
```

### 本地多GPU训练

```bash
python -m torch.distributed.run \
  --nproc_per_node=4 \
  train.py \
  --config configs/train_consistency_mini.py \
  --work-dir ./work_dirs/multi_gpu_exp
```

---

## 📈 监控训练进度

### TensorBoard（推荐）

```bash
# 安装 TensorBoard
pip install tensorboard

# 启动
tensorboard --logdir work_dirs/consistency_mini_v2/

# 浏览器访问
# http://localhost:6006
```

### 日志文件

训练日志保存在:
```
work_dirs/consistency_mini_v2/
├── checkpoints/
│   ├── latest.pth
│   └── best.pth
├── config_snapshot.json
└── eval_val_results.json  # 评估结果
```

---

## ⚠️ 常见问题

### Q1: 训练时某个维度准确率不提升？

**A**: 检查以下几点：
1. 该维度的标签是否正确构造
2. 损失权重是否太小（尝试增加）
3. 正负样本比例是否均衡

### Q2: Ranking 评估结果为空？

**A**: Ranking 评估需要索引中包含多个候选样本：
```python
# 确保索引构建时包含同一 scene 的多个样本
# 例如：不同时刻、不同扰动
```

### Q3: 显存不足？

**A**: 尝试：
```bash
# 减小 batch size
python train.py --config configs/train_consistency_mini.py --batch-size 4

# 减小图像尺寸（需要重新构建索引）
# configs/train_consistency_mini.py
image_size=128  # 从 224 降低到 128
```

---

## 🎓 下一步

完成当前训练后，建议：

1. **Phase 2**: 集成 World Model，添加 Layer 1 生成质量评估
2. **Phase 3**: 训练 inverse dynamics model，增强 Layer 2 评估
3. **Phase 4**: **核心目标** - 在 nuPlan closed-loop 中验证 critic score 与驾驶性能的相关性

**最终目标**: 证明 visual-action metric 比 FID/FVD 更能预测闭环驾驶性能。

---

## 📚 相关文档

- [README.md](README.md) - 完整项目说明
- [use.md](use.md) - 原始任务描述和详细设计
- [configs/train_consistency_mini.py](configs/train_consistency_mini.py) - 训练配置
