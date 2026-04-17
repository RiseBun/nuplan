# NuPlan Critic 框架升级总结

## 📋 升级概览

本次升级将原始的**二分类 Critic** 改造为**多维度 Action-Conditioned Consequence Evaluator**，以支持三层评估框架。

---

## 🔄 核心改动

### 1. 配置文件升级

**文件**: `configs/train_consistency_mini.py`

**改动**:
- ✅ 实验名称更新: `consistency_mini_v1` → `consistency_mini_v2`
- ✅ 新增多维度损失权重配置:
  ```python
  lambda_speed_consistency=0.3
  lambda_steering_consistency=0.3
  lambda_progress_consistency=0.2
  lambda_temporal_coherence=0.2
  ```
- ✅ 新增 Ranking 评估配置:
  ```python
  ranking=dict(
      enabled=True,
      num_candidates_per_scene=5,
      ranking_metrics=["ndcg@3", "ndcg@5", "mrr", "top1_hit_rate"],
  )
  ```

---

### 2. 模型架构升级

**文件**: `train.py` - `ConsistencyCriticModel`

**改动**:
- ✅ 从**双头**升级为**六头**评估:
  ```
  原始:
  - consistency_head
  - validity_head
  
  升级后:
  - consistency_head           (总体一致性)
  - speed_consistency_head     (速度一致性)
  - steering_consistency_head  (转向一致性)
  - progress_consistency_head  (前进一致性)
  - temporal_coherence_head    (时间连贯性)
  - validity_head              (驾驶合理性)
  ```

- ✅ 共享 Backbone 保持不变:
  - History/Future Image Encoder（共享CNN）
  - Trajectory Encoder
  - Ego Encoder

---

### 3. 训练循环升级

**文件**: `train.py` - `run_consistency_epoch()`

**改动**:
- ✅ 支持多维度损失计算:
  ```python
  loss = (lambda_c * loss_c + 
         lambda_v * loss_v + 
         lambda_speed * loss_speed +
         lambda_steering * loss_steering +
         lambda_progress * loss_progress +
         lambda_temporal * loss_temporal)
  ```

- ✅ 多维度准确率追踪:
  - `c_acc`, `v_acc`
  - `speed_acc`, `steering_acc`, `progress_acc`, `temporal_acc`

- ✅ 支持多维度标签（向后兼容）:
  ```python
  speed_labels = batch.get("speed_consistency_label", c_labels)
  steering_labels = batch.get("steering_consistency_label", c_labels)
  # ...
  ```

---

### 4. 评估脚本升级

**文件**: `eval_critic.py`

**改动**:
- ✅ 新增 `compute_ranking_metrics()` 函数:
  - NDCG@3, NDCG@5
  - MRR (Mean Reciprocal Rank)
  - Top-1 Hit Rate

- ✅ 新增 `--eval-ranking` 命令行参数

- ✅ 按 scene 分组评估排序能力

---

### 5. 文档升级

**文件**: `README.md`

**改动**:
- ✅ 新增"项目定位"章节（三层评估框架说明）
- ✅ 更新模型结构章节（多维度评估头）
- ✅ 新增 Ranking 评估能力说明
- ✅ 更新"下一步建议"（5个Phase详细规划）
- ✅ 新增"核心贡献点"章节（论文叙事框架）

**新文件**:
- ✅ `GUIDE.md` - 完整使用指南
- ✅ `UPGRADE_SUMMARY.md` - 本文件

---

## 📊 升级前后对比

| 维度 | 升级前 | 升级后 |
|------|--------|--------|
| **评估头数量** | 2 (consistency, validity) | 6 (+ speed, steering, progress, temporal) |
| **损失函数** | 单一 BCE | 多维度加权 BCE |
| **评估指标** | Accuracy, F1, AUC | + NDCG, MRR, Top-1 Hit Rate |
| **训练监控** | 2个准确率 | 6个准确率 |
| **Ranking能力** | ❌ | ✅ |
| **文档完整度** | 基础说明 | 三层框架 + 使用指南 + 升级路径 |

---

## 🎯 对齐三层评估框架

### Layer 1: 生成质量评估
- **状态**: ⚠️ 待实现（需要 World Model）
- **计划**: Phase 2 集成 FID/FVD 计算

### Layer 2: Action一致性评估
- **状态**: ✅ **已实现**
- **能力**:
  - 多维度一致性评估（speed, steering, progress, temporal）
  - Ranking 能力（NDCG, MRR, Top-1 Hit Rate）
  - 细粒度负样本支持

### Layer 3: 驾驶合理性评估
- **状态**: ✅ **已实现**
- **能力**:
  - Validity head 评估驾驶合理性
  - 支持正负样本权重调节

---

## 🚀 使用方法

### 训练

```bash
# 使用新配置训练
python train.py --config configs/train_consistency_mini.py

# 训练日志会显示6个维度的准确率
[Epoch 1/30] loss=0.6234 c_acc=0.7123 v_acc=0.6845 
             speed_acc=0.7456 steering_acc=0.7234 
             progress_acc=0.6987 temporal_acc=0.7123
```

### 评估

```bash
# 基础评估
python eval_critic.py \
  --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth

# 包含 Ranking 评估
python eval_critic.py \
  --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth \
  --eval-ranking
```

### Ranking 输出示例

```
[Ranking Metrics]
  场景数: 500
  NDCG@3:  0.7654
  NDCG@5:  0.7823
  MRR:     0.7234
  Top-1 Hit Rate: 0.6987
```

---

## 🔧 向后兼容性

### ✅ 完全向后兼容

1. **基础 Critic 模型** (`CriticModel`) 保持不变
   - 仍可使用 `configs/train_critic_mini.py` 训练

2. **多维度标签可选**
   - 如果索引中缺少多维度标签，会自动 fallback 到 `consistency_label`
   ```python
   speed_labels = batch.get("speed_consistency_label", c_labels)
   ```

3. **旧索引文件仍可用**
   - 现有 `consistency_train.jsonl` 可直接使用
   - 新维度标签为可选增强

---

## 📈 预期效果

### 训练指标更丰富

```
之前:
[Epoch 1/10] train_loss=0.6234 train_acc=0.7123 val_loss=0.5987 val_acc=0.7234

现在:
[Epoch 1/30] loss=0.6234 c_acc=0.7123 v_acc=0.6845 
             speed_acc=0.7456 steering_acc=0.7234 
             progress_acc=0.6987 temporal_acc=0.7123 
             val_loss=0.5987 val_c_acc=0.7234 val_v_acc=0.6923
```

### 评估能力更强

```
之前:
- Accuracy: 0.7234
- F1 Score: 0.7214
- AUC: 0.7891

现在:
+ Ranking Metrics:
  - NDCG@3: 0.7654
  - NDCG@5: 0.7823
  - MRR: 0.7234
  - Top-1 Hit Rate: 0.6987
```

---

## 🎓 下一步行动

### 立即可做

1. ✅ 使用新配置训练模型
2. ✅ 观察多维度准确率变化
3. ✅ 运行 Ranking 评估

### 短期计划（1-2周）

1. 构建包含多维度标签的索引
   - 基于轨迹特征生成 speed/steering/progress 标签
2. 调优多维度损失权重
3. 分析各维度准确率与总体性能的关系

### 中期计划（2-4周）

1. 集成 World Model（Phase 2）
2. 实现 FID/FVD 计算
3. 开始闭环相关性实验设计（Phase 4）

### 长期目标（1-2月）

1. **核心目标**: 证明 critic score 与 nuPlan closed-loop performance 的相关性 > FID/FVD
2. 撰写论文
3. 开源代码

---

## 📝 技术细节

### 多维度标签构造建议

如果需要构造细粒度标签：

```python
# 示例：基于轨迹特征生成标签
def construct_dimension_labels(traj, gt_traj):
    """
    traj: 候选轨迹 (B, 8, 3)
    gt_traj: 真实轨迹 (B, 8, 3)
    """
    # Speed consistency
    speed_diff = np.abs(traj[:, :, 0] - gt_traj[:, :, 0]).mean(axis=1)
    speed_label = (speed_diff < threshold_speed).astype(float)
    
    # Steering consistency
    steering_diff = np.abs(traj[:, :, 2] - gt_traj[:, :, 2]).mean(axis=1)
    steering_label = (steering_diff < threshold_steering).astype(float)
    
    # Progress consistency
    progress_diff = np.abs(traj[:, :, 0].sum(axis=1) - gt_traj[:, :, 0].sum(axis=1))
    progress_label = (progress_diff < threshold_progress).astype(float)
    
    # Temporal coherence
    temporal_smooth = np.std(np.diff(traj, axis=1), axis=1).mean(axis=1)
    temporal_label = (temporal_smooth < threshold_temporal).astype(float)
    
    return {
        "speed_consistency_label": speed_label,
        "steering_consistency_label": steering_label,
        "progress_consistency_label": progress_label,
        "temporal_coherence_label": temporal_label,
    }
```

---

## ✅ 升级检查清单

- [x] 配置文件升级（多维度权重 + Ranking配置）
- [x] 模型架构升级（6个评估头）
- [x] 训练循环升级（多维度loss + 准确率追踪）
- [x] 评估脚本升级（Ranking指标）
- [x] README 更新（三层框架说明）
- [x] 使用指南创建（GUIDE.md）
- [x] 向后兼容性保证
- [x] 文档完整性

---

## 🎉 总结

本次升级成功将框架从**简单二分类**提升到**多维度评估系统**，为三层评估框架打下坚实基础。

**核心成就**:
1. ✅ 多维度 Action 一致性评估
2. ✅ Ranking 能力评估
3. ✅ 完整的文档和使用指南
4. ✅ 清晰的后续发展路径

**下一步核心目标**: 
> 证明 visual-action metric 与 nuPlan 闭环性能的相关性高于传统 FID/FVD

这将是一个 **strong contribution**，足以支撑一篇高质量论文。
