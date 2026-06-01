# IAC：图像-动作一致性评测基准

IAC（Image-Action Consistency）是一个面向自动驾驶 World Action Model（WAM）的评测基准。它要回答的问题很简单：给定一段历史图像、一个动作或未来轨迹，以及 WAM 生成的未来图像，这些未来图像是否真的反映了这个动作？同时，这条轨迹本身是否符合基本运动学约束？

本项目不是 planner，也不是 world model。IAC 不负责生成图像或生成轨迹，只负责对已有的 WAM 输出进行打分。

## 我们的模型评测什么

IAC Critic 有两个输出头：

- `consistency`：判断“未来图像”和“候选轨迹/动作”是否一致。例如轨迹明显向左转，但未来画面仍像直行，这一项应该低分。
- `validity`：判断候选轨迹本身是否合理。例如速度突变、横向偏移过大、航向变化不连续，这一项应该低分。

最终用于 benchmark 的两个分数是：

- `iac_consistency`：图像-动作一致性分数，越高表示 WAM 生成的未来图像越符合给定动作。
- `iac_validity`：轨迹运动学合理性分数，越高表示轨迹本身越可行。

IAC 的主输出是连续分数，不建议只看固定 `0.5` 阈值。阈值属于 checkpoint 和校准验证集，不是模型无关常量。旧版 checkpoint 的示例 operating point 是：

- `consistency_threshold=0.31`：偏均衡，适合默认 pass/fail。
- `consistency_threshold=0.255`：偏高召回，适合尽量不漏掉一致样本。
- `validity_threshold=0.97`：用于更严格的轨迹合理性 pass/fail。

训练新的 checkpoint 后，必须在对应验证集上重新校准阈值。`benchmark_wam.py` 默认只输出连续分数；只有显式传入阈值时才输出 pass/fail。

## 输入是什么

训练和评测时，每个样本包含四类输入：

- `history_images`：历史相机图像序列。
- `future_images`：未来图像序列。训练时通常来自 nuPlan 真实未来帧；评测 WAM 时来自 WAM 生成帧。
- `ego_state`：当前自车状态，例如速度、加速度、yaw、yaw rate 等。
- `candidate_traj`：待评测的未来轨迹，表示一个动作或动作序列。

当前索引构建默认按时间偏移抽取历史和未来图像，而不是取相邻相机帧：

- `history_images`：`-1.5s, -1.0s, -0.5s, 0.0s`
- `future_images`：`+0.5s, +1.0s, +1.5s, +2.0s`

这样历史序列包含真实运动变化，避免连续相机帧几乎完全相同、信息量不足的问题。可以用 `--history-image-offsets` 和 `--future-image-offsets` 覆盖这两个时间窗口。

WAM benchmark 的 JSONL 输入示例：

```json
{
  "wam_name": "my_wam",
  "group_id": "scene_001",
  "history_images": ["history_0.jpg", "history_1.jpg", "history_2.jpg", "history_3.jpg"],
  "future_images": ["wam_future_0.jpg", "wam_future_1.jpg", "wam_future_2.jpg", "wam_future_3.jpg"],
  "ego_state": [0.0, 0.0, 0.0, 0.0, 0.0],
  "candidate_traj": [[0.0, 0.0, 0.0], [1.0, 0.1, 0.02]],
  "action_type": "left_turn"
}
```

`consistency_label` 和 `validity_label` 是可选字段。如果输入里提供标签，脚本会额外计算 accuracy、recall、AUROC、PR-AUC 等监督评估指标；如果没有标签，脚本仍然会输出每个 WAM 的 IAC 分数。

## 输出是什么

运行 `benchmark_wam.py` 后会生成两个文件：

```text
work_dirs/wam_benchmark/<name>/
├── wam_iac_scores.jsonl
└── wam_iac_summary.json
```

`wam_iac_scores.jsonl` 是逐样本结果，包含：

- 样本 ID / 分组 ID。
- WAM 名称。
- `iac_consistency`。
- `iac_validity`。
- 如果显式传入阈值，会包含 `consistency_pass` / `validity_pass`。
- 如果显式传入阈值，会记录使用的 calibrated threshold。
- 可选的预测标签和原始 logit。

`wam_iac_summary.json` 是汇总结果，包含：

- overall 平均分。
- 按 WAM 分组的平均分。
- 按动作类型分组的平均分。
- 如果有同一场景下的多候选动作，会计算 ranking 指标。
- 如果有扰动强度字段，会计算 graded perturbation curve。

## 我们是怎么学习的

IAC 使用自监督/弱监督构造训练样本，不需要人工标注“图像和动作是否一致”。

正样本来自真实 nuPlan 片段：

- 历史图像是真实历史帧。
- 未来图像是真实未来帧。
- 候选轨迹是真实 ego 未来轨迹。
- 因此 `consistency_label=1`，`validity_label=1`。

负样本由真实片段自动构造：

- `traj_swap`：图像不变，换成别的场景或别的时刻的轨迹。
- `image_swap`：轨迹不变，换成别的未来图像。
- `time_shift_future`：使用时间错位的未来图像。
- `perturb_lateral`：横向扰动轨迹。
- `perturb_heading`：扰动航向角。
- `perturb_speed`：扰动速度/进度。

这些负样本让模型学习“图像变化应该和动作一致”，而不是只记住图像质量或轨迹平滑度。为了降低标签噪声，当前默认构造更可见的扰动，并为每个 anchor 写入 `group_id`，让同一真实未来下的正样本和负候选可以组成 ranking 评测组。

训练时使用二分类损失作为基础监督，并额外加入同组排序损失：

```text
loss = BCE(consistency) + 0.5 * BCE(validity) + 0.2 * group_ranking_loss
```

`group_ranking_loss` 只在同一 `group_id` 内比较候选，目标是让真实匹配轨迹的 `consistency` logit 高于负候选。这和 WAM benchmark 的实际用途更一致：同一场景下，模型不仅要判断单个样本是否一致，还要把最匹配的候选动作排在前面。

因此 benchmark 不只看二分类 accuracy。更重要的是：

- 连续分数排序能力：`AUC` / `PR-AUC`。
- 同一 `group_id` 下的候选排序：`Top-1 Hit Rate` / `MRR` / `NDCG`。
- 校准阈值下的 `Recall` / `TNR`。

## 基本流程

构建 IAC 索引：

```bash
python tools/build_consistency_index.py \
  --db-root "$NUPLAN_DB_ROOT" \
  --image-roots /path/to/nuplan-v1.1_mini_camera_0 /path/to/nuplan-v1.1_mini_camera_1 \
  --output-dir indices_v4 \
  --history-image-offsets -1.5 -1.0 -0.5 0.0 \
  --future-image-offsets 0.5 1.0 1.5 2.0
```

如果你已经解压更多 nuPlan camera shard，可以把所有 camera 根目录传入 `--image-roots`，或设置 `NUPLAN_CAMERA_ROOTS`。未设置环境变量时，脚本会自动发现 `NUPLAN_DATA_ROOT` 下的全部 camera shard。默认只使用前向 `CAM_F0`，这是当前 V1 benchmark 的公平接口；多相机版本后续再扩展模型输入。

重建索引后，需要把 `NUPLAN_INDEX_ROOT` 或 `data_paths.py` 指向新的索引目录，例如 `indices_v4`，再重新训练 checkpoint。旧 checkpoint 仍然对应旧索引和旧历史帧采样规则。

训练 IAC Critic：

```bash
PYTHONUNBUFFERED=1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29619 \
  train.py \
  --config configs/train_consistency_mini.py \
  --work-dir work_dirs/iac_v4_gru_rank_2gpu_b96_w16 \
  --epochs 5 \
  --batch-size 96 \
  --num-workers 16 \
  --preflight-samples 256
```

评估 IAC Critic：

```bash
python eval_critic.py \
  --checkpoint work_dirs/iac_v4_gru_rank_2gpu_b96_w16/checkpoints/best.pth \
  --split val \
  --batch-size 128 \
  --num-workers 8 \
  --max-samples 4096 \
  --eval-ranking \
  --output-prefix epoch5_quick_val_4096_rank
```

评测 WAM 输出：

```bash
python benchmark_wam.py \
  --input path/to/wam_outputs.jsonl \
  --checkpoint work_dirs/iac_v4_gru_rank_2gpu_b96_w16/checkpoints/best.pth \
  --output-dir work_dirs/wam_benchmark/my_wam \
  --consistency-threshold 0.31 \
  --validity-threshold 0.97
```

## 数据路径

默认按 AutoDL 当前环境查找：

```text
/root/autodl-tmp/data/cache/mini
/root/autodl-tmp/nuplan-v1.1_mini_camera_0
/root/autodl-tmp/nuplan-v1.1_mini_camera_1
```

也可以用环境变量覆盖：

```bash
export NUPLAN_DATA_ROOT=/path/to/data-root
export NUPLAN_DB_ROOT=/path/to/data/cache/mini
export NUPLAN_INDEX_ROOT=/path/to/IAC/indices_v4
export NUPLAN_CAMERA_ROOTS="/path/to/camera_0:/path/to/camera_1"
```

## 仓库里保留什么

仓库只保留 IAC benchmark 主链路：

- `train.py`：训练 IAC Critic。
- `eval_critic.py`：验证 IAC Critic。
- `benchmark_wam.py`：评测 WAM 输出。
- `stress_test_iac.py`：检查模型是否依赖捷径。
- `tools/build_consistency_index.py`：构建训练/验证索引。
- `configs/train_consistency_mini.py`：默认训练配置。
- `data_paths.py`：路径配置。
- `scripts/dlc_train.sh`：训练脚本模板。

以下内容不进入仓库：

- nuPlan 原始数据和相机图像。
- 训练索引 JSONL。
- checkpoint。
- `work_dirs`。
- 训练日志。
- `__pycache__`。
- 本地 smoke/test 输出。

## DrivingWorld 是否有用

当前版本不保留 DrivingWorld 集成。原因是 IAC benchmark 的边界是“评测 WAM 输出”，而不是“在本仓库里运行某个 WAM 生成图像”。如果要评测 DrivingWorld，只需要先用 DrivingWorld 在外部生成未来图像和对应 manifest，再把 manifest 输入 `benchmark_wam.py`。这样 IAC 对 DrivingWorld、Drive-WM 或任何其它 WAM 都是同一个接口、同一套打分逻辑，更适合作为公平 benchmark。
