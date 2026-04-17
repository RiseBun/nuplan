# 🚀 DLC 多卡训练 - 快速启动指南

## ✅ 状态：已配置完成

您的 DLC 多卡训练脚本已就绪：`scripts/dlc_train.sh`

---

## 📊 训练环境配置

### 脚本位置

```
/mnt/cpfs/prediction/lipeinan/nuplan/scripts/dlc_train.sh
```

### 环境信息

| 配置项 | 值 |
|--------|-----|
| Python 路径 | `/root/anaconda3/envs/flow_planner/bin/python` |
| Conda 环境 | `flow_planner` |
| 默认 GPU 数 | 自动检测（通常 8 卡/节点） |
| NCCL 超时 | 1800 秒 |
| 文件描述符 | 65536 |

---

## 🎯 完整训练流程（DLC 多卡版）

### Phase 1: 数据准备（单卡即可）

```bash
cd /mnt/cpfs/prediction/lipeinan/nuplan

# Step 1: 生成训练数据（可以用单卡）
python generate_critic_training_data.py \
  --data-root /mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set \
  --output-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data \
  --num-scenes 500 \
  --samples-per-scene 5 \
  --device cuda:0  # 使用单卡

# Step 2: 计算标签（单卡）
python compute_training_labels.py \
  --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data \
  --output-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
  --device cuda:0

# Step 3: 构建索引（CPU）
python build_critic_index.py \
  --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
  --output-dir /mnt/cpfs/prediction/lipeinan/nuplan/indices \
  --train-ratio 0.8 \
  --balance-classes
```

**预计时间**: 5-7 小时（数据生成占大部分时间）

---

### Phase 2: 多卡训练

#### 方案 1: 单节点 8 卡训练（推荐）

```bash
# 在 DLC 平台上执行
cd /mnt/cpfs/prediction/lipeinan/nuplan

bash scripts/dlc_train.sh \
  --config configs/train_consistency_mini.py \
  --epochs 50 \
  --batch-size 16 \
  --work-dir work_dirs/critic_full_8gpu
```

**训练配置**:
- GPUs: 8 (单节点)
- Batch Size: 16 (每张卡 2 个样本)
- Effective Batch Size: 16 × 8 = 128
- 预计时间: ~3-4 小时

**监控训练**:
```bash
# 查看训练日志
tail -f work_dirs/critic_full_8gpu/logs/training.log

# 查看 GPU 使用
watch -n 1 nvidia-smi
```

---

#### 方案 2: 多节点训练（如果需要更快）

在 DLC 平台配置多节点任务：

**DLC 任务配置**:
```yaml
# DLC Web Console 或配置文件
job:
  name: "consistency_critic_multinode"
  nodes: 2              # 2 个节点
  gpus_per_node: 8      # 每节点 8 卡
  total_gpus: 16        # 总共 16 卡
  
  environment:
    - "WORLD_SIZE=2"
    - "NPROC_PER_NODE=8"
    - "MASTER_PORT=29500"
  
  command: |
    cd /mnt/cpfs/prediction/lipeinan/nuplan
    bash scripts/dlc_train.sh \
      --config configs/train_consistency_full.py \
      --epochs 50 \
      --batch-size 32 \
      --work-dir work_dirs/critic_full_16gpu
```

**训练配置**:
- GPUs: 16 (2 节点 × 8 卡)
- Batch Size: 32 (每张卡 2 个样本)
- Effective Batch Size: 32 × 16 = 512
- 预计时间: ~2 小时

---

### Phase 3: 评估

```bash
# 评估最佳模型
python eval_critic.py \
  --checkpoint work_dirs/critic_full_8gpu/checkpoints/best.pth \
  --val-index indices/consistency_val.jsonl \
  --eval-ranking \
  --output-dir eval_results/critic_full_8gpu \
  --device cuda:0
```

---

## 🔧 DLC 训练脚本详解

### 环境变量说明

脚本会自动处理以下 DLC 环境变量：

```bash
# DLC 自动注入
WORLD_SIZE=${WORLD_SIZE:-1}        # 总节点数
RANK=${RANK:-0}                    # 当前节点 rank
MASTER_ADDR=${MASTER_ADDR:-"localhost"}  # 主节点地址
MASTER_PORT=${MASTER_PORT:-29500}  # 通信端口
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # 每节点 GPU 数
```

### 关键配置

```bash
# 优化 NCCL 通信
export NCCL_TIMEOUT=1800                    # 超时 30 分钟
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # 异步错误处理

# 避免冲突
unset WORLD_SIZE    # torchrun 会重新设置
unset RANK
unset LOCAL_RANK

# 性能优化
export OMP_NUM_THREADS=4        # CPU 线程数
export PYTHONWARNINGS="ignore"  # 忽略警告

# 系统优化
ulimit -n 65536  # 提高文件描述符限制
```

### 启动命令

```bash
$PYTHON_PATH -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config $CONFIG_FILE \
    "${EXTRA_ARGS[@]}"
```

---

## 💡 训练调优建议

### 1. 批处理大小调整

根据 GPU 显存调整：

```bash
# 如果 OOM，减小 batch_size
bash scripts/dlc_train.sh \
  --batch-size 8 \
  --work-dir work_dirs/critic_bs8

# 如果显存充足，增大 batch_size
bash scripts/dlc_train.sh \
  --batch-size 24 \
  --work-dir work_dirs/critic_bs24
```

**显存参考**（每卡）:
- Batch Size 8: ~10 GB
- Batch Size 16: ~15 GB
- Batch Size 24: ~20 GB

---

### 2. 学习率调整

多卡训练时，学习率需要线性缩放：

```python
# configs/train_consistency_full.py

# 单卡基准学习率
base_lr = 1e-4

# 多卡缩放（Linear Scaling Rule）
# lr = base_lr * (total_batch_size / single_gpu_batch_size)
# 例如：8 卡，batch_size=16
# lr = 1e-4 * (16*8 / 8) = 8e-4

optimizer=dict(
    lr=8e-4,  # 8 卡训练
    # lr=16e-4,  # 16 卡训练
    weight_decay=1e-2,
)
```

---

### 3. 梯度累积（如果显存不足）

如果单卡 batch_size 太小，可以使用梯度累积：

```python
# 在 train.py 的训练循环中
accumulation_steps = 4  # 累积 4 个 step

for batch_idx, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 4. 混合精度训练（加速）

如果还没启用 AMP：

```python
# 在 train.py 中
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(batch)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📊 训练监控

### 实时查看训练进度

```bash
# 查看日志
tail -f work_dirs/critic_full_8gpu/logs/training.log

# 查看最新指标
cat work_dirs/critic_full_8gpu/metrics/metrics.json | python -m json.tool

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看 NCCL 通信
export NCCL_DEBUG=INFO
bash scripts/dlc_train.sh --config ...
```

---

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir work_dirs/critic_full_8gpu/tensorboard --port 6006

# 在浏览器中访问
# http://your-server:6006
```

---

## ⚠️ 常见问题

### Q1: CUDA OOM

**错误**: `CUDA out of memory`

**解决**:
```bash
# 减小 batch_size
bash scripts/dlc_train.sh --batch-size 8

# 或减少 num_workers
# 在配置文件中修改
num_workers=2
```

---

### Q2: NCCL 超时

**错误**: `NCCL watchdog thread terminated with exception`

**解决**:
```bash
# 脚本已设置 1800 秒超时
# 如果还不够，可以增加
export NCCL_TIMEOUT=3600

bash scripts/dlc_train.sh --config ...
```

---

### Q3: 多节点通信失败

**错误**: `Connection refused` 或 `Timeout`

**解决**:
```bash
# 检查网络连通性
ping $MASTER_ADDR

# 检查端口是否开放
nc -zv $MASTER_ADDR $MASTER_PORT

# 确保防火墙开放端口
# 或更换端口
bash scripts/dlc_train.sh \
  --config ... \
  --master-port 29501
```

---

### Q4: 数据加载慢

**现象**: GPU 利用率低（< 50%）

**解决**:
```bash
# 增加 num_workers（在配置文件中）
num_workers=8

# 或使用更多 CPU 核心
export OMP_NUM_THREADS=8
```

---

## 🎯 推荐训练配置

### 小规模验证（调试用）

```bash
bash scripts/dlc_train.sh \
  --config configs/train_consistency_mini.py \
  --epochs 10 \
  --batch-size 8 \
  --work-dir work_dirs/debug
```

**时间**: ~30 分钟  
**用途**: 验证流程、检查数据

---

### 中等规模（推荐起步）

```bash
bash scripts/dlc_train.sh \
  --config configs/train_consistency_mini.py \
  --epochs 50 \
  --batch-size 16 \
  --work-dir work_dirs/critic_medium
```

**时间**: ~3-4 小时  
**用途**: 初步验证模型效果

---

### 大规模（正式训练）

```bash
# 先构建完整训练索引
python build_critic_index.py \
  --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
  --output-dir indices \
  --num-scenes 5000 \
  --balance-classes

# 多卡训练
bash scripts/dlc_train.sh \
  --config configs/train_consistency_full.py \
  --epochs 100 \
  --batch-size 32 \
  --work-dir work_dirs/critic_large
```

**时间**: ~6-8 小时  
**用途**: 训练最终模型

---

## 📝 训练检查清单

训练前确认：

- [ ] 数据已生成（`critic_training_data/`）
- [ ] 标签已计算（`critic_training_data_labeled/`）
- [ ] 索引已构建（`indices/consistency_train.jsonl`）
- [ ] GPU 状态正常（`nvidia-smi`）
- [ ] 显存充足（至少 10 GB/卡）
- [ ] 磁盘空间充足（至少 100 GB）

训练中监控：

- [ ] GPU 利用率 > 70%
- [ ] 训练 loss 下降
- [ ] 验证 loss 稳定
- [ ] 无 OOM 错误
- [ ] NCCL 通信正常

训练后验证：

- [ ] 最佳模型已保存（`best.pth`）
- [ ] 训练日志完整
- [ ] 指标已记录
- [ ] 可以加载模型

---

## 🚀 快速开始（一键命令）

```bash
# 完整流程（从数据生成到训练完成）
cd /mnt/cpfs/prediction/lipeinan/nuplan

# 1. 生成数据（单卡，5-7 小时）
python generate_critic_training_data.py \
  --data-root /mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set \
  --output-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data \
  --num-scenes 500 \
  --samples-per-scene 5 \
  --device cuda:0

# 2. 计算标签（单卡，1 小时）
python compute_training_labels.py \
  --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data \
  --output-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
  --device cuda:0

# 3. 构建索引（CPU，1 分钟）
python build_critic_index.py \
  --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
  --output-dir indices \
  --balance-classes

# 4. 多卡训练（8 卡，3-4 小时）
bash scripts/dlc_train.sh \
  --config configs/train_consistency_mini.py \
  --epochs 50 \
  --batch-size 16 \
  --work-dir work_dirs/critic_full

# 5. 评估（单卡，30 分钟）
python eval_critic.py \
  --checkpoint work_dirs/critic_full/checkpoints/best.pth \
  --val-index indices/consistency_val.jsonl \
  --eval-ranking \
  --output-dir eval_results/critic_full
```

**总时间**: ~10-12 小时  
**GPU 需求**: 8 卡（训练阶段）

---

**🎊 DLC 多卡训练已配置完成！可以直接使用了！**
