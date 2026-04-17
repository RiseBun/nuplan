NuPlan Action–Image Consistency Critic for DrivingWorld

它的目标不是单纯判断“轨迹好不好”，而是判断：

在当前驾驶上下文下，给定未来动作/轨迹条件后，world model 生成的未来图像是否真的遵守了这个条件，并且这个未来是否在驾驶上合理。

1. 项目目标

你最终要服务的是 DrivingWorld 这类 action-conditioned world model。根据你前面的分析，DrivingWorld 本体更像是“带 pose/yaw 条件的可控视觉转移模型”，而不是完整 planner；所以你现在最需要的，不是一个普通 trajectory critic，而是一个能对 (history, action, generated future) 进行判别的 critic。

因此整个项目定义为两个子目标：

目标 A：Condition Consistency

判断：

给定未来动作/未来轨迹条件后，生成未来图像是否与该条件一致。

也就是检查：

左转条件下，图像是否真的表现出左转结果
换道条件下，图像是否真的体现车道横向迁移
减速条件下，图像光流/前向进展是否合理减弱
目标 B：Driving Validity

判断：

即使生成图像与动作一致，这个未来是否仍然是驾驶上合理、安全、可执行的。

也就是区分两种失败：

不 obey condition
obey 了 condition，但未来本身不合理

这两个目标必须分开，不然你的评测会混掉。这个区分和你前面在 world model 表征分析里强调的 “coherence” 与 “planning relevance” 的差异是完全一致的。

2. 任务形式

我建议不要只做一个二分类，而是做双头 critic。

输入统一为：

历史图像：history_images
当前 ego state：ego_state
候选未来动作/轨迹：candidate_traj
未来图像：future_images

输出两个分数：

consistency_logit
预测未来图像与动作条件是否匹配
validity_logit
预测这组未来是否驾驶合理

最终：

sigmoid(consistency_logit) = consistency probability
sigmoid(validity_logit) = driving-valid probability

还可以再定义一个组合分数：

𝑠
𝑐
𝑜
𝑟
𝑒
𝑓
𝑖
𝑛
𝑎
𝑙
=
𝛼
⋅
𝑝
𝑐
𝑜
𝑛
𝑠
𝑖
𝑠
𝑡
𝑒
𝑛
𝑐
𝑦
+
(
1
−
𝛼
)
⋅
𝑝
𝑣
𝑎
𝑙
𝑖
𝑑
𝑖
𝑡
𝑦
score
final
	​

=α⋅p
consistency
	​

+(1−α)⋅p
validity
	​


或者更保守一点：

𝑠
𝑐
𝑜
𝑟
𝑒
𝑓
𝑖
𝑛
𝑎
𝑙
=
𝑝
𝑐
𝑜
𝑛
𝑠
𝑖
𝑠
𝑡
𝑒
𝑛
𝑐
𝑦
×
𝑝
𝑣
𝑎
𝑙
𝑖
𝑑
𝑖
𝑡
𝑦
score
final
	​

=p
consistency
	​

×p
validity
	​


后者更适合闭环筛选，因为只有两者都高才通过。

3. 输入输出定义
3.1 输入

沿用你现有 critic 的输入组织方式，再补 future images。

输入 1：历史图像

建议仍然用最近 4 帧 CAM_F0：

history_images: (B, 4, 3, 224, 224)

这和你当前工程保持一致，改动最小。

输入 2：当前 ego state

仍然保留当前 5 维：

ego_state: (B, 5)
[vx, vy, yaw_rate, ax, angular_rate_z]

也可后续扩到 7~9 维，但第一版不需要。

输入 3：候选未来动作/轨迹

继续用：

candidate_traj: (B, 8, 3)
每步 (dx, dy, dyaw)
覆盖未来 4 秒，步长 0.5s

这一点非常好，因为它已经天然对应 DrivingWorld 里 future pose/yaw condition 的语义。

输入 4：未来图像

新增：

future_images: (B, 4, 3, 224, 224) 或 (B, 8, 3, 224, 224)

建议第一版先用 4 帧：

例如 future 0.5s, 1.5s, 2.5s, 4.0s
这样既覆盖短中期，又避免算力太大

如果你后续想更强，可以用 8 帧全量。

3.2 输出

建议输出：

consistency_logit: (B,)
validity_logit: (B,)

训练时两个 BCE loss。
推理时得到：

p_consistency = sigmoid(consistency_logit)
p_validity = sigmoid(validity_logit)
4. 数据定义与标签构造

这里是整套方案最关键的部分。

你当前 trajectory critic 的正负样本定义是：

正：真实 future trajectory
负：其他时间点采样的轨迹

这对于 trajectory plausibility 可以，但对于 image–action consistency 不够。

所以新数据格式要改成：

{
  "sample_id": "...",
  "scene_name": "...",
  "timestamp_us": ...,
  "history_images": [...4 paths...],
  "future_images": [...4 paths...],
  "ego_state": [...],
  "candidate_traj": [[dx, dy, dyaw], ...],
  "consistency_label": 0 or 1,
  "validity_label": 0 or 1,
  "source_type": "gt_pos | traj_swap | image_swap | perturb | model_rollout"
}
4.1 正样本
正样本 P1：GT-aligned
历史图像：真实
ego state：真实
candidate_traj：真实未来轨迹
future_images：真实未来图像
consistency_label = 1
validity_label = 1

这是最天然的正样本。

正样本 P2：Model-good rollout
历史图像：真实
ego state：真实
candidate_traj：某个给定 action
future_images：world model rollout
条件：经过规则筛选，确实和 action 对齐，且未来合理
consistency_label = 1
validity_label = 1

这一类不是第一阶段必须有，但第二阶段非常重要。
否则 critic 永远只学 GT 分布，不会真正适配 DrivingWorld。

4.2 负样本

必须做成多来源 hard negatives。

负样本 N1：Trajectory swap
history、ego、future_images 保持真实
candidate_traj 换成其他时间点/其他场景的轨迹
consistency = 0
validity = 0 or 可能未知

这是你已有方法，可保留。

负样本 N2：Future image swap
history、ego、candidate_traj 保持真实
future_images 换成其他时间点的真实未来
consistency = 0

这会强迫模型必须看 future image，而不是只看 trajectory。

负样本 N3：Trajectory perturb

对 GT future trajectory 做轻度语义扰动：

直行 → 轻微左偏 / 轻微右偏
左转 → 直行
轻刹 → 匀速
匀速 → 轻加速

future_images 仍然保持 GT。
标签：

consistency = 0
validity 取决于扰动大小，建议第一版直接也标 0

这是最有价值的一类，因为它不是“明显不对”，而是“很像，但不匹配”。

负样本 N4：Model rollout hard negative

直接从 DrivingWorld 采 rollout：

给定 action condition
生成 future_images
如果视觉上/几何上明显不 obey condition，则
consistency = 0
如果未来明显出界/异常/不合理，则
validity = 0

这是让 critic 真正和你的 world model 接轨的关键。

负样本 N5：Counterfactual pair

同一段 history，构造两条相近但不同语义的轨迹：

一条是真实 future trajectory
一条是“局部改过的 near-miss trajectory”

然后和同一个 future image 配对。
这样会形成很强的对比样本。

5. validity label 怎么定义

这是你最容易纠结的点。第一版不要追求完美“驾驶合理性真值”，用规则代理即可。

建议 validity=0 的条件满足任一条：

轨迹明显超出局部可行驶区域
轨迹曲率/位姿变化异常，超出经验上限
轨迹与当前状态严重不连续
生成 future image 出现明显崩坏
与 GT future 的 endpoint / heading 偏差过大

第一版可以偏保守，宁可把 validity 定义得粗一点。

因为你的主战场其实还是 consistency。
validity 只要能提供第二个维度，已经足够有价值。

6. 模型结构

你现在的 CriticModel 很轻量，这很好。第一版不要上大 transformer。建议做四支编码器 + 双头融合。

6.1 模型总体结构
HistoryImageEncoder   -> z_hist   (256)
FutureImageEncoder    -> z_future (256)
TrajectoryEncoder     -> z_traj   (128)
EgoEncoder            -> z_ego    (128)

Concat -> z_all (768)

Shared Fusion MLP -> z_shared (256)

Head 1: ConsistencyHead -> 1
Head 2: ValidityHead    -> 1
6.2 各模块建议
A. HistoryImageEncoder

保留你现在的 4 层 CNN + pooling 思路。
因为历史图像主要是提供上下文，不一定要太强。

可做法：

对每帧单独 CNN
然后 temporal average pooling / attention pooling
输出 256-d
B. FutureImageEncoder

和 history encoder 结构共享或不共享都行。

建议第一版共享 backbone，不共享最后投影层。
理由：

历史图像主要看上下文
未来图像主要看结果
backbone 共享可省参数，head 不共享保留角色差异
C. TrajectoryEncoder

继续 2 层 MLP 即可。
但建议对轨迹增加一个增强表示：

原始 (dx, dy, dyaw) 之外，再额外算：

cumulative forward progress
cumulative lateral shift
final heading change
mean curvature / approximate curvature

然后拼接后再送入 MLP。
这样轨迹 encoder 更容易抓语义。

D. EgoEncoder

原样保留。

E. Fusion

先共享，再双头。
不要一开始就完全分成两个独立模型，否则数据效率差。

7. 训练目标

总损失：

𝐿
=
𝜆
𝑐
𝐿
𝑐
𝑜
𝑛
𝑠
𝑖
𝑠
𝑡
𝑒
𝑛
𝑐
𝑦
+
𝜆
𝑣
𝐿
𝑣
𝑎
𝑙
𝑖
𝑑
𝑖
𝑡
𝑦
L=λ
c
	​

L
consistency
	​

+λ
v
	​

L
validity
	​


其中两项都用 BCEWithLogitsLoss。

初始建议：

lambda_c = 1.0
lambda_v = 0.5

因为 consistency 是主任务。

8. 数据集构建流程

你原来有：

NuPlan SQLite + camera images
    -> build_critic_index.py
    -> critic_train.jsonl / critic_val.jsonl

现在改成：

NuPlan SQLite + camera images
    -> build_consistency_index.py
    -> base_gt_samples.jsonl
    -> augment_swap_perturb.py
    -> critic_consistency_train.jsonl / val.jsonl
8.1 第一步：抽 GT 对齐样本

对每个 anchor time：

取最近 4 帧历史图像
取未来 4 帧图像
取当前 ego state
取未来 8-step GT trajectory

得到 P1 正样本。

8.2 第二步：构造错配样本

针对每个 P1：

构造 1 个 traj swap
构造 1 个 image swap
构造 1~2 个 perturb
可选再加 1 个 intra-scene hard negative

这样每个正样本对应 3~5 个负样本。

建议第一版控制比例：

正:负 = 1:2 或 1:3

不要过多，否则容易学成“全负”。

8.3 第三步：加入 model rollout 样本

等你先把基础 critic 训稳，再把 DrivingWorld rollout 数据加入训练集：

存成和 GT 同格式 JSONL
source_type = model_rollout
可以单独混入 10%~20%

这样 critic 会真正认识模型分布，而不是只认识 nuPlan GT 分布。

9. 训练策略
第一阶段：GT-supervised pretrain

只用 GT 正样本 + 构造负样本训练。

目标：

学会最基本的 image-action matching
学会区分 plausibility 与 mismatch
第二阶段：Model-adaptation finetune

加入 DrivingWorld rollout 样本微调。

目标：

减少 domain gap
让 critic 对你的生成分布更敏感
10. 评估指标

不能只报 accuracy。要分任务。

10.1 Consistency 指标
Accuracy
Precision
Recall
F1
ROC-AUC
正负样本平均概率差
10.2 Validity 指标

同上。

10.3 排序指标

给同一 history 下多条 candidate trajectories / rollouts：

Top-1 hit rate
Top-k hit rate
MRR
NDCG

这类指标最能说明 critic 能否做候选重排序。

10.4 闭环指标

用 critic 参与 rollout 选择后，比较：

endpoint error
heading error
lane-change obey rate
turn obey rate
collision proxy
off-road proxy

即使你还不能做严格闭环 simulator，这些 proxy 也够用。

11. 闭环使用方式

你要服务 DrivingWorld，所以真正推理时流程应是：

Step 1

给定当前 history 和 ego state

Step 2

生成多条 candidate future actions / trajectories
来源可以是：

GT perturb
预定义模板
planner proposal
DrivingWorld 自己条件采样
Step 3

DrivingWorld 对每条 candidate rollout 出 future images

Step 4

critic 对每条 (history, ego, candidate_traj, future_images) 打分

输出：

p_consistency
p_validity
p_final
Step 5

选分数最高的一条

这才是完整闭环。

12. 为什么这个方案比你当前版本更对

你当前版本本质上是：

history-conditioned trajectory plausibility classifier

而我给你的这版是：

action-conditioned future image–trajectory consistency critic with driving validity awareness

两者区别非常大：

你当前版能回答
这条轨迹像不像一个合理未来
新方案能回答
给定这个动作条件，生成出来的未来图像有没有 obey
即使 obey，这个未来是否驾驶合理
哪条 rollout 更值得选

这正好对上你前面对 DrivingWorld 的判断：它的重点是“给定控制后的视觉响应”，而不是策略选择本身。你的 critic 就应该围绕这个点来建。

13. 最容易踩的坑
坑 1：负样本太假

如果只做跨场景随机 swap，模型很容易学会“场景差异”，而不是 consistency。

解决：

多做同场景 hard negatives
多做微小 perturb
多做 counterfactual pair
坑 2：future image 分支太强

模型可能只看生成图像是否自然，不看 action。

解决：
做消融实验：

去掉 trajectory 输入
去掉 future image 输入
history + traj only
history + future only
full model

只有 full model 明显最好，才能证明它真的学了 image-action consistency。

坑 3：validity 标签噪声大

第一版不要太激进，规则粗一点也没关系。
只要 consistency head 足够强，整个项目就成立。

坑 4：critic 只会识别 GT，不会识别 world model

所以第二阶段一定要用 model rollout 微调。

14. 一个最小可跑版本

如果你现在想尽快起步，我建议只做下面这版：

数据
history_images: 4 帧
future_images: 4 帧
ego_state: 5 维
candidate_traj: 8x3
标签
只做 consistency_label
暂时不做 validity
负样本
traj swap
image swap
perturb
模型
4 支 encoder
1 个 shared fusion
1 个 consistency head
评估
Accuracy / F1 / AUC
同 history 多 candidate 的 rank accuracy

这版最容易先跑通。

然后第二阶段再加：

validity head
rollout 微调
闭环排序实验
15. 推荐的项目目录
nuplan_consistency_critic/
├── configs/
│   ├── train_consistency_mini.py
│   └── train_consistency_full.py
├── data/
│   ├── critic_consistency_train.jsonl
│   ├── critic_consistency_val.jsonl
│   └── sample_stats.json
├── tools/
│   ├── build_consistency_index.py
│   ├── augment_negative_pairs.py
│   ├── build_model_rollout_index.py
│   └── visualize_samples.py
├── models/
│   ├── image_encoder.py
│   ├── future_image_encoder.py
│   ├── trajectory_encoder.py
│   ├── ego_encoder.py
│   └── critic_model.py
├── train.py
├── eval.py
├── eval_ranking.py
├── eval_closed_loop_proxy.py
└── work_dirs/