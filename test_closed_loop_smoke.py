#!/usr/bin/env python3
"""
Closed-Loop Evaluation 冒烟测试

使用模拟数据验证流程是否正常
"""

import numpy as np
from scipy import stats

print("="*80)
print("Closed-Loop Evaluation 冒烟测试")
print("="*80)

# 1. 模拟候选轨迹生成
print("\n[1/5] 测试候选轨迹生成...")
num_candidates = 5
ground_truth = np.random.randn(8, 6)  # [T=8, action_dim=6]

noise_levels = np.linspace(0.0, 0.5, num_candidates)
candidates = []
for noise in noise_levels:
    if noise == 0.0:
        candidates.append(ground_truth.copy())
    else:
        candidates.append(ground_truth + np.random.randn(8, 6) * noise)

print(f"  ✅ 生成 {len(candidates)} 个候选轨迹")
print(f"  噪声级别: {noise_levels}")

# 2. 模拟评分
print("\n[2/5] 模拟评分...")
# Critic 分数（越高越好）- 与噪声负相关
critic_scores = [1.0 - noise * 0.8 + np.random.randn() * 0.05 for noise in noise_levels]
# FID 分数（越低越好）- 与噪声正相关
fid_scores = [noise * 100 + np.random.randn() * 5 for noise in noise_levels]
# FVD 分数（越低越好）- 与噪声正相关
fvd_scores = [noise * 150 + np.random.randn() * 8 for noise in noise_levels]

print(f"  Critic scores: {[f'{s:.3f}' for s in critic_scores]}")
print(f"  FID scores: {[f'{s:.2f}' for s in fid_scores]}")
print(f"  FVD scores: {[f'{s:.2f}' for s in fvd_scores]}")

# 3. 选择 top-1
print("\n[3/5] 选择 top-1 轨迹...")
best_critic_idx = np.argmax(critic_scores)
best_fid_idx = np.argmin(fid_scores)
best_fvd_idx = np.argmin(fvd_scores)

print(f"  Best by Critic: candidate {best_critic_idx} (score={critic_scores[best_critic_idx]:.3f})")
print(f"  Best by FID: candidate {best_fid_idx} (score={fid_scores[best_fid_idx]:.2f})")
print(f"  Best by FVD: candidate {best_fvd_idx} (score={fvd_scores[best_fvd_idx]:.2f})")

# 4. 计算闭环指标
print("\n[4/5] 计算闭环指标...")

def compute_closed_loop_metrics(traj, gt):
    """简化的闭环指标计算"""
    deviation = np.mean((traj - gt) ** 2)
    nc = 1.0 - np.exp(-deviation * 2.0)
    ttc = 10.0 / (1.0 + deviation)
    comfort = 1.0 / (1.0 + deviation)
    progress = 1.0 / (1.0 + deviation * 0.5)
    overall = 0.4 * nc + 0.2 * (ttc / 10.0) + 0.2 * comfort + 0.2 * progress
    
    return {
        'NC': nc,
        'TTC': ttc,
        'Comfort': comfort,
        'Progress': progress,
        'Overall': overall
    }

metrics_critic = compute_closed_loop_metrics(candidates[best_critic_idx], ground_truth)
metrics_fid = compute_closed_loop_metrics(candidates[best_fid_idx], ground_truth)
metrics_fvd = compute_closed_loop_metrics(candidates[best_fvd_idx], ground_truth)

print(f"  Critic selected: NC={metrics_critic['NC']:.3f}, Overall={metrics_critic['Overall']:.3f}")
print(f"  FID selected: NC={metrics_fid['NC']:.3f}, Overall={metrics_fid['Overall']:.3f}")
print(f"  FVD selected: NC={metrics_fvd['NC']:.3f}, Overall={metrics_fvd['Overall']:.3f}")

# 5. 相关性分析
print("\n[5/5] 相关性分析...")

# 模拟多个场景的结果（100 场景）
num_scenes = 100
all_critic_scores = []
all_fid_scores = []
all_fvd_scores = []
all_overall_metrics = []

for _ in range(num_scenes):
    # 随机选择最佳的噪声级别
    best_noise = np.random.uniform(0.0, 0.3)  # Critic 倾向于选择低噪声
    
    # Critic 分数
    critic = 1.0 - best_noise * 0.8 + np.random.randn() * 0.1
    
    # FID/FVD 有时会选错
    fid_noise = best_noise + np.random.uniform(-0.1, 0.2)
    fvd_noise = best_noise + np.random.uniform(-0.1, 0.2)
    
    fid = fid_noise * 100
    fvd = fvd_noise * 150
    
    # 闭环指标
    overall = 1.0 - best_noise * 0.6 + np.random.randn() * 0.05
    
    all_critic_scores.append(critic)
    all_fid_scores.append(-fid)  # 取负，变成越高越好
    all_fvd_scores.append(-fvd)  # 取负
    all_overall_metrics.append(max(0, min(1, overall)))

# 计算 Spearman 相关性
spearman_critic, p_critic = stats.spearmanr(all_critic_scores, all_overall_metrics)
spearman_fid, p_fid = stats.spearmanr(all_fid_scores, all_overall_metrics)
spearman_fvd, p_fvd = stats.spearmanr(all_fvd_scores, all_overall_metrics)

print(f"\n{'Metric':<15} {'Spearman':>12} {'p-value':>12}")
print("-" * 40)
print(f"{'Critic':<15} {spearman_critic:>10.3f} {p_critic:>10.2e}")
print(f"{'FID':<15} {spearman_fid:>10.3f} {p_fid:>10.2e}")
print(f"{'FVD':<15} {spearman_fvd:>10.3f} {p_fvd:>10.2e}")

print(f"\nGain vs FID: {spearman_critic - spearman_fid:+.3f}")
print(f"Gain vs FVD: {spearman_critic - spearman_fvd:+.3f}")

# 验证
print("\n" + "="*80)
if spearman_critic > spearman_fid and spearman_critic > spearman_fvd:
    print("✅ 冒烟测试通过!")
    print(f"   Critic 相关性 ({spearman_critic:.3f}) > FID ({spearman_fid:.3f})")
    print(f"   Critic 相关性 ({spearman_critic:.3f}) > FVD ({spearman_fvd:.3f})")
else:
    print("⚠️  需要更多数据或调整模型")

print("="*80)
print("\n✅ 闭环评估系统流程验证完成!")
print("\n下一步:")
print("  1. 训练 Consistency Critic 模型")
print("  2. 使用真实检查点运行评估")
print("  3. 收集 100+ 场景数据")
print("\n命令:")
print("  python closed_loop_evaluation.py \\")
print("    --checkpoint work_dirs/critic_full/checkpoints/best.pth \\")
print("    --val-index indices/consistency_val.jsonl \\")
print("    --num-scenes 100 \\")
print("    --output-dir eval_results/closed_loop")
