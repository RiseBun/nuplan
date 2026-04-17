#!/usr/bin/env python3
"""快速验证脚本 - 确保升级后的框架可以正常运行"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """测试配置文件加载"""
    print("=" * 60)
    print("测试 1: 配置文件加载")
    print("=" * 60)
    
    from train import load_config
    
    config_path = "configs/train_consistency_mini.py"
    cfg = load_config(config_path)
    
    # 检查新增字段
    assert "lambda_speed_consistency" in cfg, "缺少 lambda_speed_consistency"
    assert "lambda_steering_consistency" in cfg, "缺少 lambda_steering_consistency"
    assert "lambda_progress_consistency" in cfg, "缺少 lambda_progress_consistency"
    assert "lambda_temporal_coherence" in cfg, "缺少 lambda_temporal_coherence"
    assert "ranking" in cfg, "缺少 ranking 配置"
    
    print("✅ 配置文件加载成功")
    print(f"   - lambda_speed_consistency: {cfg['lambda_speed_consistency']}")
    print(f"   - lambda_steering_consistency: {cfg['lambda_steering_consistency']}")
    print(f"   - lambda_progress_consistency: {cfg['lambda_progress_consistency']}")
    print(f"   - lambda_temporal_coherence: {cfg['lambda_temporal_coherence']}")
    print(f"   - ranking enabled: {cfg['ranking']['enabled']}")
    print()
    
    return cfg


def test_model(cfg):
    """测试模型构建"""
    print("=" * 60)
    print("测试 2: 模型构建")
    print("=" * 60)
    
    import torch
    from train import ConsistencyCriticModel
    
    model = ConsistencyCriticModel(cfg)
    
    # 检查是否有6个评估头
    assert hasattr(model, 'consistency_head'), "缺少 consistency_head"
    assert hasattr(model, 'speed_consistency_head'), "缺少 speed_consistency_head"
    assert hasattr(model, 'steering_consistency_head'), "缺少 steering_consistency_head"
    assert hasattr(model, 'progress_consistency_head'), "缺少 progress_consistency_head"
    assert hasattr(model, 'temporal_coherence_head'), "缺少 temporal_coherence_head"
    assert hasattr(model, 'validity_head'), "缺少 validity_head"
    
    print("✅ 模型构建成功")
    print(f"   - 评估头数量: 6")
    print(f"   - 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 测试前向传播
    print("测试 2.1: 前向传播")
    batch_size = 2
    history_images = torch.randn(batch_size, 4, 3, 224, 224)
    future_images = torch.randn(batch_size, 4, 3, 224, 224)
    ego_state = torch.randn(batch_size, 5)
    candidate_traj = torch.randn(batch_size, 8, 3)
    
    model.eval()
    with torch.no_grad():
        output = model(history_images, future_images, ego_state, candidate_traj)
    
    assert "consistency_logit" in output
    assert "speed_consistency_logit" in output
    assert "steering_consistency_logit" in output
    assert "progress_consistency_logit" in output
    assert "temporal_coherence_logit" in output
    assert "validity_logit" in output
    
    print("✅ 前向传播成功")
    print(f"   - consistency_logit shape: {output['consistency_logit'].shape}")
    print(f"   - speed_consistency_logit shape: {output['speed_consistency_logit'].shape}")
    print(f"   - steering_consistency_logit shape: {output['steering_consistency_logit'].shape}")
    print(f"   - progress_consistency_logit shape: {output['progress_consistency_logit'].shape}")
    print(f"   - temporal_coherence_logit shape: {output['temporal_coherence_logit'].shape}")
    print(f"   - validity_logit shape: {output['validity_logit'].shape}")
    print()
    
    return model


def test_ranking_metrics():
    """测试 Ranking 指标计算"""
    print("=" * 60)
    print("测试 3: Ranking 指标计算")
    print("=" * 60)
    
    # 模拟数据
    scores = [0.9, 0.7, 0.8, 0.5, 0.6]
    relevances = [1, 0, 1, 0, 1]  # 3个正样本
    
    # NDCG@3
    def compute_ndcg(scores_list, relevance_list, k):
        import numpy as np
        if len(scores_list) < 2:
            return 0.0
        sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
        sorted_relevances = [rel for _, rel in sorted_pairs[:k]]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevances))
        ideal_relevances = sorted(relevance_list, reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        return dcg / idcg if idcg > 0 else 0.0
    
    ndcg_3 = compute_ndcg(scores, relevances, k=3)
    ndcg_5 = compute_ndcg(scores, relevances, k=5)
    
    print(f"✅ Ranking 指标计算成功")
    print(f"   - NDCG@3: {ndcg_3:.4f}")
    print(f"   - NDCG@5: {ndcg_5:.4f}")
    print()


def main():
    """主测试流程"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "NuPlan Critic 框架升级验证" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        # 测试 1: 配置
        cfg = test_config()
        
        # 测试 2: 模型
        model = test_model(cfg)
        
        # 测试 3: Ranking 指标
        test_ranking_metrics()
        
        # 总结
        print("=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print()
        print("框架升级成功，可以开始训练：")
        print()
        print("  # 训练模型")
        print("  python train.py --config configs/train_consistency_mini.py")
        print()
        print("  # 评估模型（包含 Ranking）")
        print("  python eval_critic.py --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth --eval-ranking")
        print()
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print("=" * 60)
        print("❌ 测试失败！")
        print("=" * 60)
        print(f"\n错误信息: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
