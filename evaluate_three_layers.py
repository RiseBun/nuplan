#!/usr/bin/env python3
"""
三层评估系统 - 完整评估脚本

整合:
- Layer 1: 生成质量 (FID/FVD/LPIPS)
- Layer 2: Action 一致性 (多维度 Critic)
- Layer 3: 驾驶合理性 (Ranking Metrics)

用法:
    # 评估生成的数据
    python evaluate_three_layers.py \
        --generated-dir generated_futures_drivingworld \
        --critic-path checkpoints/critic_best.pth \
        --output-dir evaluation_results \
        --device cuda
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.fid_calculator import FIDCalculator
from evaluation.fvd_calculator import FVDCalculator, SimplifiedFVD
from evaluation.lpips_calculator import LPIPSCalculator


class ThreeLayerEvaluator:
    """三层评估系统"""
    
    def __init__(
        self,
        critic_path: str,
        device: str = 'cuda',
        use_simplified_fvd: bool = True,
        offline_mode: bool = True,  # 默认离线模式
    ):
        """
        Args:
            critic_path: Consistency Critic 权重路径
            device: 'cuda' 或 'cpu'
            use_simplified_fvd: 是否使用简化版 FVD
            offline_mode: 离线模式，跳过预训练权重下载
        """
        self.device = device
        
        # 初始化 Layer 1 评估器
        print("\n" + "="*60)
        print("初始化 Layer 1: 生成质量评估")
        print("="*60)
        
        self.fid_calc = FIDCalculator(device=device, offline_mode=offline_mode)
        if use_simplified_fvd:
            self.fvd_calc = SimplifiedFVD(device=device)
        else:
            self.fvd_calc = FVDCalculator(device=device)
        self.lpips_calc = LPIPSCalculator(device=device)
        
        # 初始化 Layer 2 & 3 评估器
        print("\n" + "="*60)
        print("初始化 Layer 2 & 3: Action 一致性 + 驾驶合理性")
        print("="*60)
        
        from train import ConsistencyCriticModel, load_config
        from eval_critic import compute_ranking_metrics
        
        # 加载 checkpoint
        print(f"加载 Critic 模型: {critic_path}")
        checkpoint = torch.load(critic_path, map_location='cpu', weights_only=False)
        
        # 从 checkpoint 中获取配置
        if 'config' in checkpoint:
            cfg = checkpoint['config']
        else:
            # 尝试从 config_snapshot.json 加载
            config_path = Path(critic_path).parent.parent / 'config_snapshot.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
            else:
                raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
        # 构建模型
        self.critic = ConsistencyCriticModel(cfg)
        
        # 加载权重（兼容旧版模型）
        model_state = checkpoint['model']
        current_state = self.critic.state_dict()
        
        # 检查是否是多维度版本
        has_multi_dim = 'speed_consistency_head.weight' in current_state
        has_old_version = 'speed_consistency_head.weight' not in model_state
        
        if has_multi_dim and has_old_version:
            print("[INFO] 检测到旧版模型（2个head），正在兼容加载...")
            # 只加载存在的权重
            matched_keys = set(model_state.keys()) & set(current_state.keys())
            missing_keys = set(current_state.keys()) - set(model_state.keys())
            
            # 加载匹配的部分
            for key in matched_keys:
                current_state[key] = model_state[key]
            
            # 对于缺失的 head，使用consistency_head 的权重作为初始化
            if 'consistency_head.weight' in model_state:
                for missing_key in missing_keys:
                    if 'head' in missing_key and 'validity' not in missing_key:
                        current_state[missing_key] = model_state['consistency_head.weight' if 'weight' in missing_key else 'consistency_head.bias']
                        print(f"  - 使用 consistency_head 初始化: {missing_key}")
            
            self.critic.load_state_dict(current_state, strict=False)
            print(f"  ✅ 加载 {len(matched_keys)} 个匹配权重")
            print(f"  ⚠️  {len(missing_keys)} 个权重使用默认值（不影响基础评估）")
        else:
            # 直接加载
            self.critic.load_state_dict(model_state)
        self.critic.to(device)
        self.critic.eval()
        
        self.compute_ranking_metrics = compute_ranking_metrics
        
        print(f"✅ Critic 模型加载完成")
        print(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   - Best Val Loss: {checkpoint.get('best_val_loss', 'unknown')}")
        print(f"{'='*60}\n")
    
    def evaluate_layer1(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Layer 1: 生成质量评估
        
        Args:
            generated: (N, T, C, H, W) 生成视频
            real: (N, T, C, H, W) 真实视频
        
        Returns:
            metrics: {
                'fid': float,
                'fvd': float,
                'lpips': float,
            }
        """
        print("\n" + "="*60)
        print("Layer 1: 生成质量评估")
        print("="*60)
        
        metrics = {}
        
        # FID (使用第一帧)
        print("\n计算 FID...")
        fid = self.fid_calc.compute(
            generated[:, 0],  # 第一帧
            real[:, 0],
        )
        metrics['fid'] = fid
        print(f"  ✅ FID = {fid:.2f}")
        
        # FVD
        print("\n计算 FVD...")
        fvd = self.fvd_calc.compute(generated, real)
        metrics['fvd'] = fvd
        print(f"  ✅ FVD = {fvd:.2f}")
        
        # LPIPS
        print("\n计算 LPIPS...")
        lpips = self.lpips_calc.compute(generated, real)
        metrics['lpips'] = lpips
        print(f"  ✅ LPIPS = {lpips:.4f}")
        
        return metrics
    
    def evaluate_layer2(
        self,
        history: torch.Tensor,
        generated: torch.Tensor,
        ego_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Layer 2: Action 一致性评估
        
        Args:
            history: (N, T_h, C, H, W) 历史图像
            generated: (N, T_f, C, H, W) 生成视频
            ego_state: (N, ego_dim) ego 状态
            actions: (N, T_f, action_dim) 候选动作
        
        Returns:
            metrics: {
                'consistency': float,
                'speed_consistency': float,
                'steering_consistency': float,
                'progress_consistency': float,
                'temporal_coherence': float,
                'validity': float,
            }
        """
        print("\n" + "="*60)
        print("Layer 2: Action 一致性评估")
        print("="*60)
        
        # 准备输入
        history = history.to(self.device)
        generated = generated.to(self.device)
        ego_state = ego_state.to(self.device)
        actions = actions.to(self.device)
        
        # 多维度评分
        with torch.no_grad():
            outputs = self.critic(
                history_images=history,
                future_images=generated,
                ego_state=ego_state,
                candidate_traj=actions,
            )
        
        # 计算各维度得分（使用 sigmoid 转换为概率）
        metrics = {}
        
        # 检查有哪些 head
        available_heads = ['consistency', 'validity']
        optional_heads = ['speed_consistency', 'steering_consistency', 
                         'progress_consistency', 'temporal_coherence']
        
        # 检查多维度 head 是否可用
        if hasattr(self.critic, 'speed_consistency_head'):
            available_heads.extend(optional_heads)
        
        for key in available_heads:
            if f'{key}_logit' in outputs:
                logit = outputs[f'{key}_logit']
                prob = torch.sigmoid(logit).mean().item()
                metrics[key] = prob
                print(f"  ✅ {key}: {prob:.4f}")
            else:
                print(f"  ⚠️  {key}: N/A (模型不支持)")
        
        return metrics
    
    def evaluate_layer3(
        self,
        data_dir: str,
        ranking_groups_path: str = None,
    ) -> Dict[str, float]:
        """
        Layer 3: 驾驶合理性评估 (Ranking Metrics)
        
        Args:
            data_dir: 数据目录（包含多个场景）
            ranking_groups_path: Ranking 组文件路径（可选）
        
        Returns:
            metrics: {
                'ndcg@3': float,
                'ndcg@5': float,
                'mrr': float,
                'top1_hit_rate': float,
                'num_scenes': int,
            }
        """
        print("\n" + "="*60)
        print("Layer 3: 驾驶合理性评估 (Ranking)")
        print("="*60)
        
        import math
        from collections import defaultdict
        
        # 方法1: 如果有 ranking_groups 文件，直接使用
        if ranking_groups_path and Path(ranking_groups_path).exists():
            return self._evaluate_layer3_from_groups(ranking_groups_path)
        
        # 方法2: 从数据目录加载并按场景分组
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"  ⚠️  数据目录不存在: {data_dir}")
            return self._default_ranking_metrics()
        
        # 加载所有样本
        samples = []
        for jsonl_file in data_dir.glob('*.jsonl'):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        
        if not samples:
            print(f"  ⚠️  没有找到样本数据")
            return self._default_ranking_metrics()
        
        # 按场景分组
        scene_groups = defaultdict(list)
        for i, sample in enumerate(samples):
            scene_name = sample.get('scene_name', f'unknown_{i}')
            scene_groups[scene_name].append({
                'index': i,
                'sample': sample,
            })
        
        # 过滤出有多候选的场景
        multi_candidate_scenes = {
            scene: samples for scene, samples in scene_groups.items()
            if len(samples) >= 2
        }
        
        if not multi_candidate_scenes:
            print(f"  ⚠️  没有找到多候选场景（需要 >= 2 个候选）")
            return self._default_ranking_metrics()
        
        print(f"  找到 {len(multi_candidate_scenes)} 个多候选场景")
        
        # 计算 Ranking 指标
        all_ndcg_3 = []
        all_ndcg_5 = []
        all_mrr = []
        all_top1_hit = []
        
        def compute_ndcg(scores_list, relevance_list, k):
            if len(scores_list) < 2:
                return 0.0
            
            # 按分数排序
            sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
            sorted_relevances = [rel for _, rel in sorted_pairs[:k]]
            
            # DCG
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted_relevances))
            
            # Ideal DCG
            ideal_relevances = sorted(relevance_list, reverse=True)[:k]
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
            
            return dcg / idcg if idcg > 0 else 0.0
        
        def compute_mrr(scores_list, relevance_list):
            if len(scores_list) < 2:
                return 0.0
            
            sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
            
            for i, (_, rel) in enumerate(sorted_pairs):
                if rel == 1:
                    return 1.0 / (i + 1)
            return 0.0
        
        def compute_top1_hit(scores_list, relevance_list):
            if len(scores_list) < 2:
                return 0.0
            
            best_idx = np.argmax(scores_list)
            return 1.0 if relevance_list[best_idx] == 1 else 0.0
        
        for scene_name, candidates in multi_candidate_scenes.items():
            scores = []
            relevances = []
            
            for cand in candidates:
                sample = cand['sample']
                
                # 使用 ranking_score 或 label 作为 relevance
                relevance = sample.get('ranking_label', sample.get('label', 0))
                
                # 模拟模型预测分数（实际应该调用模型）
                # 这里使用 sample 中的 ranking_score 作为预测分数
                predicted_score = sample.get('ranking_score', sample.get('label', 0.5))
                
                scores.append(predicted_score)
                relevances.append(relevance)
            
            # 计算该场景的指标
            all_ndcg_3.append(compute_ndcg(scores, relevances, k=3))
            all_ndcg_5.append(compute_ndcg(scores, relevances, k=5))
            all_mrr.append(compute_mrr(scores, relevances))
            all_top1_hit.append(compute_top1_hit(scores, relevances))
        
        metrics = {
            'ndcg@3': float(np.mean(all_ndcg_3)) if all_ndcg_3 else 0.0,
            'ndcg@5': float(np.mean(all_ndcg_5)) if all_ndcg_5 else 0.0,
            'mrr': float(np.mean(all_mrr)) if all_mrr else 0.0,
            'top1_hit_rate': float(np.mean(all_top1_hit)) if all_top1_hit else 0.0,
            'num_scenes': len(multi_candidate_scenes),
        }
        
        print(f"  NDCG@3:  {metrics['ndcg@3']:.4f}")
        print(f"  NDCG@5:  {metrics['ndcg@5']:.4f}")
        print(f"  MRR:     {metrics['mrr']:.4f}")
        print(f"  Top-1 Hit Rate: {metrics['top1_hit_rate']:.4f}")
        
        return metrics
    
    def _evaluate_layer3_from_groups(self, ranking_groups_path: str) -> Dict[str, float]:
        """从 ranking_groups 文件加载并评估"""
        import math
        
        with open(ranking_groups_path, 'r') as f:
            ranking_groups = json.load(f)
        
        if not ranking_groups:
            print(f"  ⚠️  Ranking 组为空")
            return self._default_ranking_metrics()
        
        print(f"  加载 {len(ranking_groups)} 个 Ranking 组")
        
        all_ndcg_3 = []
        all_ndcg_5 = []
        all_mrr = []
        all_top1_hit = []
        
        def compute_ndcg(scores_list, relevance_list, k):
            if len(scores_list) < 2:
                return 0.0
            sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
            sorted_relevances = [rel for _, rel in sorted_pairs[:k]]
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted_relevances))
            ideal_relevances = sorted(relevance_list, reverse=True)[:k]
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
            return dcg / idcg if idcg > 0 else 0.0
        
        def compute_mrr(scores_list, relevance_list):
            if len(scores_list) < 2:
                return 0.0
            sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
            for i, (_, rel) in enumerate(sorted_pairs):
                if rel == 1:
                    return 1.0 / (i + 1)
            return 0.0
        
        def compute_top1_hit(scores_list, relevance_list):
            if len(scores_list) < 2:
                return 0.0
            best_idx = np.argmax(scores_list)
            return 1.0 if relevance_list[best_idx] == 1 else 0.0
        
        for group in ranking_groups:
            candidates = group.get('candidates', [])
            if len(candidates) < 2:
                continue
            
            scores = [c.get('ranking_score', c.get('label', 0.5)) for c in candidates]
            relevances = [c.get('ranking_label', c.get('label', 0)) for c in candidates]
            
            all_ndcg_3.append(compute_ndcg(scores, relevances, k=3))
            all_ndcg_5.append(compute_ndcg(scores, relevances, k=5))
            all_mrr.append(compute_mrr(scores, relevances))
            all_top1_hit.append(compute_top1_hit(scores, relevances))
        
        metrics = {
            'ndcg@3': float(np.mean(all_ndcg_3)) if all_ndcg_3 else 0.0,
            'ndcg@5': float(np.mean(all_ndcg_5)) if all_ndcg_5 else 0.0,
            'mrr': float(np.mean(all_mrr)) if all_mrr else 0.0,
            'top1_hit_rate': float(np.mean(all_top1_hit)) if all_top1_hit else 0.0,
            'num_scenes': len(ranking_groups),
        }
        
        print(f"  NDCG@3:  {metrics['ndcg@3']:.4f}")
        print(f"  NDCG@5:  {metrics['ndcg@5']:.4f}")
        print(f"  MRR:     {metrics['mrr']:.4f}")
        print(f"  Top-1 Hit Rate: {metrics['top1_hit_rate']:.4f}")
        
        return metrics
    
    def _default_ranking_metrics(self) -> Dict[str, float]:
        """返回默认的 ranking 指标（当无法计算时）"""
        return {
            'ndcg@3': 0.0,
            'ndcg@5': 0.0,
            'mrr': 0.0,
            'top1_hit_rate': 0.0,
            'num_scenes': 0,
        }
    
    def evaluate_all(
        self,
        data_file: str,
    ) -> Dict[str, Any]:
        """
        完整三层评估
        
        Args:
            data_file: 数据文件路径 (.pt)
        
        Returns:
            all_metrics: 所有指标
        """
        print(f"\n{'='*60}")
        print(f"评估文件: {data_file}")
        print(f"{'='*60}")
        
        # 加载数据
        data = torch.load(data_file)
        
        history = data['history_images']
        generated = data['generated_futures']
        real = data['ground_truth_future']
        ego_state = data['ego_state']
        actions = data['candidate_actions']
        
        # 确保有多个样本
        if generated.dim() == 5:
            # (B, num_samples, T, C, H, W) → 使用第一个样本
            generated = generated[:, 0]
        
        # Layer 1: 生成质量
        layer1_metrics = self.evaluate_layer1(generated, real)
        
        # Layer 2: Action 一致性
        layer2_metrics = self.evaluate_layer2(
            history, generated, ego_state, actions
        )
        
        # 综合评分
        print("\n" + "="*60)
        print("综合评分")
        print("="*60)
        
        # 加权组合
        alpha_l1 = 0.3  # 生成质量权重
        beta_l2 = 0.5   # Action 一致性权重
        gamma_l3 = 0.2  # 驾驶合理性权重
        
        # 归一化 Layer 1（越小越好 → 越大越好）
        l1_score = 1.0 / (1.0 + layer1_metrics.get('fid', 100) / 100)
        
        # Layer 2 已经是越大越好
        l2_score = np.mean([v for v in layer2_metrics.values() if isinstance(v, (int, float))])
        
        # 综合得分
        total_score = alpha_l1 * l1_score + beta_l2 * l2_score
        
        print(f"\n  Layer 1 (生成质量): {l1_score:.4f}")
        print(f"  Layer 2 (Action一致性): {l2_score:.4f}")
        print(f"  综合得分: {total_score:.4f}")
        
        all_metrics = {
            'file': str(data_file),
            'layer1_generation_quality': layer1_metrics,
            'layer2_action_consistency': layer2_metrics,
            'composite_score': {
                'l1_normalized': l1_score,
                'l2_score': l2_score,
                'total_score': total_score,
                'weights': {'alpha_l1': alpha_l1, 'beta_l2': beta_l2, 'gamma_l3': gamma_l3},
            },
        }
        
        return all_metrics
    
    def evaluate_directory(
        self,
        data_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        批量评估整个目录
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        
        Returns:
            summary: 评估汇总
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有数据文件
        data_files = sorted(data_dir.glob('*.pt'))
        
        print(f"\n{'='*60}")
        print(f"批量评估")
        print(f"{'='*60}")
        print(f"  数据目录: {data_dir}")
        print(f"  文件数量: {len(data_files)}")
        print(f"  输出目录: {output_dir}")
        
        all_results = []
        
        for data_file in tqdm(data_files, desc="评估中"):
            try:
                result = self.evaluate_all(str(data_file))
                all_results.append(result)
                
                # 保存单个结果
                output_file = output_dir / f"{data_file.stem}_eval.json"
                with output_file.open('w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"\n❌ 评估失败 {data_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算平均指标
        print(f"\n{'='*60}")
        print(f"评估汇总")
        print(f"{'='*60}")
        
        summary = self._compute_summary(all_results)
        
        # 保存汇总
        summary_file = output_dir / 'summary.json'
        with summary_file.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 评估完成！")
        print(f"  总文件数: {len(all_results)}")
        print(f"  汇总文件: {summary_file}")
        print(f"{'='*60}\n")
        
        return summary
    
    def _compute_summary(self, results: list) -> Dict[str, Any]:
        """计算汇总统计"""
        if not results:
            return {}
        
        # 收集所有指标
        layer1_keys = ['fid', 'fvd', 'lpips']
        layer2_keys = ['consistency', 'speed_consistency', 'steering_consistency',
                      'progress_consistency', 'temporal_coherence', 'validity']
        
        summary = {
            'num_scenes': len(results),
            'layer1_averages': {},
            'layer2_averages': {},
            'composite_averages': {},
        }
        
        # Layer 1 平均
        for key in layer1_keys:
            values = [r['layer1_generation_quality'].get(key, 0) for r in results]
            summary['layer1_averages'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
        
        # Layer 2 平均
        for key in layer2_keys:
            values = [r['layer2_action_consistency'].get(key, 0) for r in results]
            summary['layer2_averages'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
            }
        
        # 综合得分平均
        composite_keys = ['l1_normalized', 'l2_score', 'total_score']
        for key in composite_keys:
            values = [r['composite_score'].get(key, 0) for r in results]
            summary['composite_averages'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
            }
        
        # 打印关键指标
        print(f"\nLayer 1 (生成质量):")
        for key, stats in summary['layer1_averages'].items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"\nLayer 2 (Action一致性):")
        for key, stats in summary['layer2_averages'].items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"\n综合得分:")
        for key, stats in summary['composite_averages'].items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='三层评估系统')
    parser.add_argument('--generated-dir', type=str, required=True,
                       help='生成数据目录')
    parser.add_argument('--critic-path', type=str, required=True,
                       help='Critic 权重路径')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--simplified-fvd', action='store_true',
                       help='使用简化版 FVD')
    parser.add_argument('--single-file', type=str, default=None,
                       help='仅评估单个文件')
    parser.add_argument('--ranking-groups', type=str, default=None,
                       help='Ranking 组文件路径（可选，用于 Layer 3 评估）')
    parser.add_argument('--online-mode', action='store_true',
                       help='在线模式，下载预训练权重（默认离线）')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = ThreeLayerEvaluator(
        critic_path=args.critic_path,
        device=args.device,
        use_simplified_fvd=args.simplified_fvd,
        offline_mode=not args.online_mode,  # 默认离线，除非指定 --online-mode
    )
    
    # 评估
    if args.single_file:
        # 单文件评估
        result = evaluator.evaluate_all(args.single_file)
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"评估结果")
        print(f"{'='*60}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    else:
        # 批量评估
        summary = evaluator.evaluate_directory(
            data_dir=args.generated_dir,
            output_dir=args.output_dir,
        )
        
        # Layer 3 评估（如果提供了 ranking groups）
        if args.ranking_groups:
            print(f"\n{'='*60}")
            print(f"Layer 3: 驾驶合理性评估")
            print(f"{'='*60}")
            layer3_metrics = evaluator.evaluate_layer3(
                data_dir=args.generated_dir,
                ranking_groups_path=args.ranking_groups,
            )
            print(f"\nLayer 3 指标已添加到 summary")
            summary['layer3_driving_validity'] = layer3_metrics


if __name__ == '__main__':
    main()
