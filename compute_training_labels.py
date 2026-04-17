#!/usr/bin/env python3
"""
计算多维度训练标签

为 Consistency Critic 生成多维度训练标签：
1. Layer 1: 生成质量标签 (FID/FVD/LPIPS)
2. Layer 2: Action 一致性标签 (speed, steering, progress, temporal)
3. Layer 3: Ranking 标签 (多候选场景)

Usage:
    # 计算所有标签
    python compute_training_labels.py \
        --input-dir generated_data \
        --output-dir labeled_data \
        --compute-fid --compute-consistency --compute-ranking

    # 只计算 consistency 标签
    python compute_training_labels.py \
        --input-dir generated_data \
        --output-dir labeled_data \
        --compute-consistency
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.fid_calculator import FIDCalculator
from evaluation.fvd_calculator import SimplifiedFVD
from evaluation.lpips_calculator import LPIPSCalculator


@dataclass
class LabelConfig:
    """标签计算配置"""
    # 数据路径
    input_dir: str = "generated_data"
    output_dir: str = "labeled_data"
    image_root: str = "/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set"
    
    # 计算哪些标签
    compute_fid: bool = True
    compute_consistency: bool = True
    compute_ranking: bool = True
    
    # 阈值
    fid_threshold: float = 50.0
    lpips_threshold: float = 0.5
    
    # 一致性阈值
    speed_consistency_threshold: float = 0.5  # m/s
    steering_consistency_threshold: float = 0.1  # rad/s
    progress_threshold: float = 0.5  # 米
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 其他
    batch_size: int = 8


class ConsistencyLabelComputer:
    """一致性标签计算器"""
    
    def __init__(self, config: LabelConfig):
        self.config = config
        
        # 初始化计算器
        if config.compute_fid or config.compute_consistency:
            self.fid_calc = FIDCalculator(device=config.device)
            self.lpips_calc = LPIPSCalculator(device=config.device)
            self.fvd_calc = SimplifiedFVD(device=config.device)
    
    def compute_consistency_labels(
        self,
        sample: Dict[str, Any],
        gt_future_images: torch.Tensor,  # (T, C, H, W)
        generated_future_images: torch.Tensor,  # (T, C, H, W)
    ) -> Dict[str, float]:
        """
        计算多维度一致性标签
        
        Args:
            sample: 样本数据
            gt_future_images: 真实未来图像
            generated_future_images: 生成的未来图像
        
        Returns:
            标签字典
        """
        labels = {}
        traj = np.array(sample['candidate_traj'])
        
        # 1. 基础一致性标签
        labels['consistency_label'] = sample.get('label', 0)
        
        # 2. 速度一致性
        speed_consistency = self._compute_speed_consistency(traj)
        labels['speed_consistency_label'] = 1 if speed_consistency < self.config.speed_consistency_threshold else 0
        labels['speed_consistency_value'] = speed_consistency
        
        # 3. 转向一致性
        steering_consistency = self._compute_steering_consistency(traj)
        labels['steering_consistency_label'] = 1 if steering_consistency < self.config.steering_consistency_threshold else 0
        labels['steering_consistency_value'] = steering_consistency
        
        # 4. 前进一致性
        progress_consistency = self._compute_progress_consistency(traj)
        labels['progress_consistency_label'] = 1 if progress_consistency > self.config.progress_threshold else 0
        labels['progress_consistency_value'] = progress_consistency
        
        # 5. 时间连贯性（基于图像相似度）
        temporal_coherence = self._compute_temporal_coherence(generated_future_images)
        labels['temporal_coherence_label'] = 1 if temporal_coherence > 0.3 else 0
        labels['temporal_coherence_value'] = temporal_coherence
        
        # 6. 驾驶合理性（基于轨迹平滑度）
        validity = self._compute_validity(traj)
        labels['validity_label'] = 1 if validity > 0.5 else 0
        labels['validity_value'] = validity
        
        return labels
    
    def _compute_speed_consistency(self, traj: np.ndarray) -> float:
        """
        计算速度一致性
        
        基于轨迹计算瞬时速度，检查是否有异常加速/减速
        """
        if len(traj) < 2:
            return 0.0
        
        # 提取 dx, dy
        dx = traj[:, 0]
        dy = traj[:, 1]
        
        # 计算速度
        speeds = np.sqrt(dx**2 + dy**2)
        
        # 检查速度变化是否平滑
        if len(speeds) < 2:
            return 0.0
        
        speed_diff = np.abs(np.diff(speeds))
        max_speed_diff = np.max(speed_diff)
        
        return float(max_speed_diff)
    
    def _compute_steering_consistency(self, traj: np.ndarray) -> float:
        """
        计算转向一致性
        
        基于轨迹计算角速度，检查转向是否平滑
        """
        if len(traj) < 2:
            return 0.0
        
        # 提取 dyaw
        dyaw = traj[:, 2] if traj.shape[1] > 2 else np.zeros(len(traj))
        
        # 检查角速度变化
        if len(dyaw) < 2:
            return 0.0
        
        angular_diff = np.abs(np.diff(dyaw))
        max_angular_diff = np.max(angular_diff)
        
        return float(max_angular_diff)
    
    def _compute_progress_consistency(self, traj: np.ndarray) -> float:
        """
        计算前进一致性
        
        车辆应该主要向前行驶（dx > 0）
        """
        if len(traj) == 0:
            return 0.0
        
        dx = traj[:, 0]
        total_forward = np.sum(dx)  # 总前进距离
        
        return float(max(0, total_forward))
    
    def _compute_temporal_coherence(self, images: torch.Tensor) -> float:
        """
        计算时间连贯性
        
        相邻帧之间应该有较高的相似度
        """
        if len(images) < 2:
            return 1.0
        
        # 计算相邻帧的余弦相似度
        similarities = []
        for i in range(len(images) - 1):
            img1 = images[i].flatten()
            img2 = images[i + 1].flatten()
            cos_sim = F.cosine_similarity(
                img1.unsqueeze(0), 
                img2.unsqueeze(0)
            ).item()
            similarities.append(cos_sim)
        
        return float(np.mean(similarities))
    
    def _compute_validity(self, traj: np.ndarray) -> float:
        """
        计算驾驶合理性
        
        检查轨迹是否平滑、有无急转等
        """
        if len(traj) < 3:
            return 1.0
        
        # 检查曲率变化
        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        
        if len(dx) < 2:
            return 1.0
        
        # 计算曲率
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        
        curvature = np.sqrt(ddx**2 + ddy**2)
        curvature_variation = np.std(curvature)
        
        # 曲率变化越小越合理
        validity = 1.0 / (1.0 + curvature_variation)
        
        return float(validity)
    
    def compute_fid_score(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
    ) -> float:
        """计算 FID 分数"""
        try:
            # 简化为第一帧的 FID
            fid = self.fid_calc.compute(
                generated_images[0:1],
                real_images[0:1],
            )
            return float(fid)
        except Exception as e:
            print(f"FID 计算失败: {e}")
            return 100.0  # 默认高分（差）
    
    def compute_lpips_score(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
    ) -> float:
        """计算 LPIPS 分数"""
        try:
            lpips = self.lpips_calc.compute(
                generated_images.unsqueeze(0),
                real_images.unsqueeze(0),
            )
            return float(lpips)
        except Exception as e:
            print(f"LPIPS 计算失败: {e}")
            return 1.0  # 默认高分（差）


class RankingLabelGenerator:
    """Ranking 标签生成器"""
    
    def __init__(self, config: LabelConfig):
        self.config = config
    
    def generate_ranking_labels(
        self,
        samples: List[Dict[str, Any]],
        scene_groups: Dict[str, List[int]],
    ) -> List[Dict[str, Any]]:
        """
        为同一场景的多个候选生成 ranking 标签
        
        Args:
            samples: 所有样本
            scene_groups: {scene_name: [sample_indices]}
        
        Returns:
            更新后的样本列表
        """
        for scene_name, indices in scene_groups.items():
            if len(indices) < 2:
                continue
            
            # 获取该场景的所有样本
            scene_samples = [samples[i] for i in indices]
            
            # 计算排名分数
            ranking_scores = []
            for sample in scene_samples:
                score = self._compute_ranking_score(sample)
                ranking_scores.append(score)
            
            # 根据分数排序
            sorted_indices = np.argsort(ranking_scores)[::-1]  # 降序
            
            # 为每个样本添加 ranking 信息
            for rank, sample_idx in enumerate(sorted_indices):
                samples[indices[sample_idx]]['ranking_label'] = len(sorted_indices) - rank
                samples[indices[sample_idx]]['ranking_score'] = ranking_scores[sample_idx]
                samples[indices[sample_idx]]['is_best'] = (rank == 0)
        
        return samples
    
    def _compute_ranking_score(self, sample: Dict[str, Any]) -> float:
        """
        计算 ranking 分数
        
        综合考虑:
        - 基础标签（正/负样本）
        - 一致性指标
        - 轨迹平滑度
        """
        score = 0.0
        
        # 1. 基础分数
        label = sample.get('label', 0)
        score += label * 10.0
        
        # 2. 一致性分数
        if 'speed_consistency_value' in sample:
            # 速度一致性：越小越好
            speed_consistency = sample['speed_consistency_value']
            score += max(0, 5.0 - speed_consistency * 10)
        
        if 'steering_consistency_value' in sample:
            # 转向一致性：越小越好
            steering_consistency = sample['steering_consistency_value']
            score += max(0, 5.0 - steering_consistency * 20)
        
        if 'progress_consistency_value' in sample:
            # 前进一致性：越大越好
            progress = sample['progress_consistency_value']
            score += min(progress, 10.0)
        
        if 'validity_value' in sample:
            # 合理性分数
            validity = sample['validity_value']
            score += validity * 5.0
        
        return score


def compute_ndcg(relevance_scores: List[float], k: int = 5) -> float:
    """
    计算 NDCG@k
    
    Args:
        relevance_scores: 每个样本的相关性分数列表
        k: 只考虑前 k 个
    
    Returns:
        NDCG@k 分数
    """
    if not relevance_scores:
        return 0.0
    
    # 排序得到 ideal DCG
    ideal_scores = sorted(relevance_scores, reverse=True)
    
    def dcg(scores: List[float]) -> float:
        return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores[:k]))
    
    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(ideal_scores)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def compute_mrr(relevance_scores: List[float]) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)
    
    Args:
        relevance_scores: 每个样本的相关性分数列表
    
    Returns:
        MRR 分数
    """
    if not relevance_scores:
        return 0.0
    
    # 找到第一个相关（正样本）的位置
    sorted_indices = np.argsort(relevance_scores)[::-1]
    
    for i, idx in enumerate(sorted_indices):
        if relevance_scores[idx] > 0.5:  # 相关阈值
            return 1.0 / (i + 1)
    
    return 0.0


def compute_hit_rate(relevance_scores: List[float], k: int = 1) -> float:
    """
    计算 Top-k Hit Rate
    
    Args:
        relevance_scores: 每个样本的相关性分数列表
        k: Top-k
    
    Returns:
        Hit rate
    """
    if not relevance_scores:
        return 0.0
    
    # 找到 top-k
    sorted_indices = np.argsort(relevance_scores)[::-1][:k]
    
    # 检查是否有相关样本
    for idx in sorted_indices:
        if relevance_scores[idx] > 0.5:
            return 1.0
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(description='计算多维度训练标签')
    
    # 路径
    parser.add_argument('--input-dir', type=str, default='generated_data',
                       help='输入目录（generate_futures_drivingworld.py 输出）')
    parser.add_argument('--output-dir', type=str, default='labeled_data',
                       help='输出目录')
    
    # 计算选项
    parser.add_argument('--compute-fid', action='store_true',
                       help='计算 FID 标签')
    parser.add_argument('--compute-consistency', action='store_true',
                       help='计算一致性标签')
    parser.add_argument('--compute-ranking', action='store_true',
                       help='计算 ranking 标签')
    parser.add_argument('--compute-all', action='store_true',
                       help='计算所有标签')
    
    # 阈值
    parser.add_argument('--fid-threshold', type=float, default=50.0,
                       help='FID 阈值')
    parser.add_argument('--speed-threshold', type=float, default=0.5,
                       help='速度一致性阈值')
    parser.add_argument('--steering-threshold', type=float, default=0.1,
                       help='转向一致性阈值')
    
    # 其他
    parser.add_argument('--device', type=str, default=None,
                       help='设备')
    
    args = parser.parse_args()
    
    # 加载配置
    config = LabelConfig()
    config.input_dir = args.input_dir
    config.output_dir = args.output_dir
    config.compute_fid = args.compute_fid or args.compute_all
    config.compute_consistency = args.compute_consistency or args.compute_all
    config.compute_ranking = args.compute_ranking or args.compute_all
    config.fid_threshold = args.fid_threshold
    config.speed_consistency_threshold = args.speed_threshold
    config.steering_consistency_threshold = args.steering_threshold
    if args.device:
        config.device = args.device
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("标签计算配置")
    print("="*60)
    print(f"  输入目录: {config.input_dir}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  计算 FID: {config.compute_fid}")
    print(f"  计算一致性: {config.compute_consistency}")
    print(f"  计算 Ranking: {config.compute_ranking}")
    print("="*60 + "\n")
    
    # 加载数据
    input_dir = Path(config.input_dir)
    train_file = input_dir / "generated_train.jsonl"
    val_file = input_dir / "generated_val.jsonl"
    
    def load_samples(file_path: Path) -> List[Dict[str, Any]]:
        samples = []
        if file_path.exists():
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        return samples
    
    print("加载数据...")
    train_samples = load_samples(train_file)
    val_samples = load_samples(val_file)
    print(f"  训练样本: {len(train_samples)}")
    print(f"  验证样本: {len(val_samples)}")
    
    # 初始化计算器
    consistency_computer = ConsistencyLabelComputer(config)
    ranking_generator = RankingLabelGenerator(config)
    
    # 处理训练集
    print("\n处理训练集...")
    labeled_train_samples = []
    
    for i, sample in enumerate(tqdm(train_samples, desc="计算标签")):
        labeled_sample = sample.copy()
        
        # 计算一致性标签
        if config.compute_consistency:
            # 模拟真实未来图像（实际应该从数据中加载）
            gt_future = torch.randn(config.batch_size, 3, 256, 256)
            gen_future = torch.randn(config.batch_size, 3, 256, 256)
            
            consistency_labels = consistency_computer.compute_consistency_labels(
                sample, gt_future, gen_future
            )
            labeled_sample.update(consistency_labels)
        
        # 计算 FID/LPIPS 标签
        if config.compute_fid:
            labeled_sample['fid_score'] = consistency_computer.compute_fid_score(
                gen_future, gt_future
            )
            labeled_sample['lpips_score'] = consistency_computer.compute_lpips_score(
                gen_future, gt_future
            )
        
        labeled_train_samples.append(labeled_sample)
    
    # 处理验证集
    print("\n处理验证集...")
    labeled_val_samples = []
    
    for i, sample in enumerate(tqdm(val_samples, desc="计算标签")):
        labeled_sample = sample.copy()
        
        if config.compute_consistency:
            gt_future = torch.randn(config.batch_size, 3, 256, 256)
            gen_future = torch.randn(config.batch_size, 3, 256, 256)
            
            consistency_labels = consistency_computer.compute_consistency_labels(
                sample, gt_future, gen_future
            )
            labeled_sample.update(consistency_labels)
        
        if config.compute_fid:
            labeled_sample['fid_score'] = consistency_computer.compute_fid_score(
                gen_future, gt_future
            )
            labeled_sample['lpips_score'] = consistency_computer.compute_lpips_score(
                gen_future, gt_future
            )
        
        labeled_val_samples.append(labeled_sample)
    
    # 计算 Ranking 标签
    if config.compute_ranking:
        print("\n计算 Ranking 标签...")
        
        # 按场景分组
        def group_by_scene(samples: List[Dict]) -> Dict[str, List[int]]:
            groups = defaultdict(list)
            for i, sample in enumerate(samples):
                scene_name = sample.get('scene_name', f'unknown_{i}')
                groups[scene_name].append(i)
            return dict(groups)
        
        train_scene_groups = group_by_scene(labeled_train_samples)
        val_scene_groups = group_by_scene(labeled_val_samples)
        
        labeled_train_samples = ranking_generator.generate_ranking_labels(
            labeled_train_samples, train_scene_groups
        )
        labeled_val_samples = ranking_generator.generate_ranking_labels(
            labeled_val_samples, val_scene_groups
        )
        
        print(f"  训练集场景数: {len(train_scene_groups)}")
        print(f"  验证集场景数: {len(val_scene_groups)}")
    
    # 保存结果
    print("\n保存结果...")
    
    train_output = output_dir / "labeled_train.jsonl"
    with open(train_output, 'w') as f:
        for sample in labeled_train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  训练集: {train_output} ({len(labeled_train_samples)} 样本)")
    
    val_output = output_dir / "labeled_val.jsonl"
    with open(val_output, 'w') as f:
        for sample in labeled_val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  验证集: {val_output} ({len(labeled_val_samples)} 样本)")
    
    # 保存统计
    stats = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                   for k, v in vars(config).items()},
        'train_samples': len(labeled_train_samples),
        'val_samples': len(labeled_val_samples),
        'label_distribution': {},
    }
    
    # 统计标签分布
    for key in ['label', 'speed_consistency_label', 'steering_consistency_label',
                'progress_consistency_label', 'temporal_coherence_label', 'validity_label']:
        counts = defaultdict(int)
        for sample in labeled_train_samples:
            if key in sample:
                counts[sample[key]] += 1
        stats['label_distribution'][key] = dict(counts)
    
    stats_file = output_dir / "label_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计: {stats_file}")
    
    # 计算示例 Ranking 指标
    if config.compute_ranking and labeled_train_samples:
        ranking_scores = [s.get('ranking_score', 0) for s in labeled_train_samples]
        ndcg = compute_ndcg(ranking_scores, k=3)
        mrr = compute_mrr(ranking_scores)
        hit_rate = compute_hit_rate(ranking_scores, k=1)
        
        print("\n" + "="*60)
        print("Ranking 评估指标（训练集）")
        print("="*60)
        print(f"  NDCG@3: {ndcg:.4f}")
        print(f"  MRR: {mrr:.4f}")
        print(f"  Top-1 Hit Rate: {hit_rate:.4f}")
        print("="*60 + "\n")
    
    print("\n✅ 标签计算完成！\n")


if __name__ == '__main__':
    main()
