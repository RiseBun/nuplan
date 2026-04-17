#!/usr/bin/env python3
"""
构建 Consistency Critic 训练索引

从 labeled_data 构建训练/验证索引，格式与 train.py 兼容。

支持:
1. 多维度一致性标签
2. Ranking 组
3. 数据增强

Usage:
    # 构建索引
    python build_consistency_index.py \
        --input-dir labeled_data \
        --output-dir indices_consistency \
        --train-ratio 0.8

    # 使用已有 nuplan 数据构建
    python build_consistency_index.py \
        --mini-db-root /path/to/mini/db \
        --image-root /path/to/images \
        --output-dir indices_consistency
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

import numpy as np
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class IndexConfig:
    """索引构建配置"""
    # 输入数据
    input_dir: str = "labeled_data"  # compute_training_labels.py 输出
    mini_db_root: str = "/mnt/datasets/e2e-datasets/20260227/e2e-datasets/dataset_pkgs/nuplan-v1.1/splits/mini"
    image_root: str = "/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set"
    
    # 输出
    output_dir: str = "indices_consistency"
    
    # 数据分割
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    random_seed: int = 42
    
    # 采样参数
    sample_stride: int = 5  # 采样间隔
    history_num_frames: int = 4  # 历史帧数
    future_steps: int = 8  # 未来步数
    future_step_time_s: float = 0.5  # 未来时间间隔
    
    # 相机配置
    camera_channel: str = "CAM_F0"
    
    # 负样本策略
    negative_sample_types: List[str] = field(default_factory=lambda: [
        'perturb_lateral', 'perturb_heading', 'perturb_speed', 
        'traj_swap', 'image_swap'
    ])
    num_negatives_per_positive: int = 4  # 每个正样本对应多少负样本
    
    # 数据增强
    enable_augmentation: bool = True
    augment_types: List[str] = field(default_factory=lambda: [
        'flip_horizontal', 'color_jitter'
    ])


class IndexBuilder:
    """索引构建器"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def build_from_labeled_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        从 labeled_data 构建索引
        
        Returns:
            (train_samples, val_samples)
        """
        input_dir = Path(self.config.input_dir)
        train_file = input_dir / "labeled_train.jsonl"
        val_file = input_dir / "labeled_val.jsonl"
        
        def load_samples(file_path: Path) -> List[Dict]:
            samples = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
            return samples
        
        # 加载数据
        train_samples = load_samples(train_file)
        val_samples = load_samples(val_file)
        
        print(f"加载 labeled 数据:")
        print(f"  训练样本: {len(train_samples)}")
        print(f"  验证样本: {len(val_samples)}")
        
        # 转换为标准格式
        train_formatted = [self._format_sample(s) for s in train_samples]
        val_formatted = [self._format_sample(s) for s in val_samples]
        
        return train_formatted, val_formatted
    
    def _format_sample(self, sample: Dict) -> Dict:
        """将 sample 格式化为 train.py 需要的格式"""
        formatted = {
            'sample_id': sample.get('sample_id', f"sample_{random.randint(0, 1000000)}"),
            'scene_name': sample.get('scene_name', 'unknown'),
            'timestamp_us': sample.get('timestamp_us', 0),
            'history_images': sample.get('history_images', []),
            'ego_state': sample.get('ego_state', [0.0] * 5),
            'candidate_traj': sample.get('candidate_traj', []),
            'label': sample.get('label', 0),
            'sample_type': sample.get('sample_type', 'unknown'),
        }
        
        # 添加多维度标签
        multi_dim_labels = [
            'speed_consistency_label',
            'steering_consistency_label',
            'progress_consistency_label',
            'temporal_coherence_label',
            'validity_label',
        ]
        for key in multi_dim_labels:
            if key in sample:
                formatted[key] = sample[key]
        
        # 添加 ranking 信息
        if 'ranking_label' in sample:
            formatted['ranking_label'] = sample['ranking_label']
        if 'ranking_score' in sample:
            formatted['ranking_score'] = sample['ranking_score']
        
        # 添加 FID/LPIPS 分数
        if 'fid_score' in sample:
            formatted['fid_score'] = sample['fid_score']
        if 'lpips_score' in sample:
            formatted['lpips_score'] = sample['lpips_score']
        
        return formatted
    
    def build_ranking_groups(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """
        构建 ranking 组
        
        将同一场景的多个候选样本打包成一个 ranking 组
        """
        # 按场景分组
        scene_groups = defaultdict(list)
        for i, sample in enumerate(samples):
            scene_name = sample.get('scene_name', f'unknown_{i}')
            scene_groups[scene_name].append(i)
        
        # 构建 ranking 组
        ranking_groups = []
        for scene_name, indices in scene_groups.items():
            if len(indices) >= 2:  # 至少需要 2 个候选
                group_samples = [samples[i] for i in indices]
                
                # 按 ranking_score 排序
                group_samples.sort(
                    key=lambda s: s.get('ranking_score', s.get('label', 0)),
                    reverse=True
                )
                
                ranking_group = {
                    'group_id': f"{scene_name}_ranking",
                    'scene_name': scene_name,
                    'num_candidates': len(group_samples),
                    'candidates': group_samples,
                }
                ranking_groups.append(ranking_group)
        
        return ranking_groups
    
    def save_index(
        self,
        samples: List[Dict],
        output_path: Path,
        include_ranking_groups: bool = True,
    ):
        """保存索引到 JSONL 文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"保存索引: {output_path} ({len(samples)} 样本)")
    
    def save_summary(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        output_dir: Path,
    ):
        """保存索引构建摘要"""
        # 统计标签分布
        def count_labels(samples: List[Dict]) -> Dict:
            counts = defaultdict(int)
            for s in samples:
                label = s.get('label', 'unknown')
                counts[label] += 1
                
                sample_type = s.get('sample_type', 'unknown')
                counts[sample_type] = counts.get(sample_type, 0) + 1
            
            return dict(counts)
        
        # 统计场景
        train_scenes = set(s.get('scene_name', '') for s in train_samples)
        val_scenes = set(s.get('scene_name', '') for s in val_samples)
        
        summary = {
            'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None), list)) else v 
                       for k, v in vars(self.config).items()},
            'num_train_samples': len(train_samples),
            'num_val_samples': len(val_samples),
            'num_train_scenes': len(train_scenes),
            'num_val_scenes': len(val_scenes),
            'train_label_distribution': count_labels(train_samples),
            'val_label_distribution': count_labels(val_samples),
            'train_scenes': sorted(list(train_scenes))[:10],  # 只保存前 10 个
            'val_scenes': sorted(list(val_scenes))[:10],
        }
        
        summary_file = output_dir / "consistency_index_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"保存摘要: {summary_file}")
        
        return summary
    
    def save_ranking_groups(
        self,
        train_groups: List[Dict],
        val_groups: List[Dict],
        output_dir: Path,
    ):
        """保存 ranking 组"""
        train_file = output_dir / "ranking_groups_train.json"
        val_file = output_dir / "ranking_groups_val.json"
        
        with open(train_file, 'w') as f:
            json.dump(train_groups, f, indent=2, ensure_ascii=False)
        
        with open(val_file, 'w') as f:
            json.dump(val_groups, f, indent=2, ensure_ascii=False)
        
        print(f"保存 Ranking 组:")
        print(f"  训练集: {train_file} ({len(train_groups)} 组)")
        print(f"  验证集: {val_file} ({len(val_groups)} 组)")


def main():
    parser = argparse.ArgumentParser(description='构建 Consistency Critic 训练索引')
    
    # 输入
    parser.add_argument('--input-dir', type=str, default='labeled_data',
                       help='labeled 数据目录')
    parser.add_argument('--mini-db-root', type=str, default=None,
                       help='nuPlan mini db 根目录')
    parser.add_argument('--image-root', type=str, default=None,
                       help='图像根目录')
    
    # 输出
    parser.add_argument('--output-dir', type=str, default='indices_consistency',
                       help='输出目录')
    
    # 分割
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 采样
    parser.add_argument('--sample-stride', type=int, default=5,
                       help='采样间隔')
    parser.add_argument('--history-frames', type=int, default=4,
                       help='历史帧数')
    parser.add_argument('--future-steps', type=int, default=8,
                       help='未来步数')
    
    # 相机
    parser.add_argument('--camera-channel', type=str, default='CAM_F0',
                       help='相机通道')
    
    args = parser.parse_args()
    
    # 加载配置
    config = IndexConfig()
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.mini_db_root:
        config.mini_db_root = args.mini_db_root
    if args.image_root:
        config.image_root = args.image_root
    if args.output_dir:
        config.output_dir = args.output_dir
    config.train_ratio = args.train_ratio
    config.random_seed = args.seed
    config.sample_stride = args.sample_stride
    config.history_num_frames = args.history_frames
    config.future_steps = args.future_steps
    config.camera_channel = args.camera_channel
    
    print("\n" + "="*60)
    print("索引构建配置")
    print("="*60)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建索引
    builder = IndexBuilder(config)
    
    # 从 labeled_data 构建
    train_samples, val_samples = builder.build_from_labeled_data()
    
    print(f"\n索引构建完成:")
    print(f"  训练样本: {len(train_samples)}")
    print(f"  验证样本: {len(val_samples)}")
    
    # 构建 Ranking 组
    print("\n构建 Ranking 组...")
    train_ranking_groups = builder.build_ranking_groups(train_samples)
    val_ranking_groups = builder.build_ranking_groups(val_samples)
    
    # 保存索引
    print("\n保存索引...")
    builder.save_index(
        train_samples, 
        output_dir / "consistency_train.jsonl"
    )
    builder.save_index(
        val_samples,
        output_dir / "consistency_val.jsonl"
    )
    
    # 保存摘要
    summary = builder.save_summary(train_samples, val_samples, output_dir)
    
    # 保存 Ranking 组
    builder.save_ranking_groups(train_ranking_groups, val_ranking_groups, output_dir)
    
    # 打印统计
    print("\n" + "="*60)
    print("索引统计")
    print("="*60)
    print(f"  训练样本: {summary['num_train_samples']}")
    print(f"  验证样本: {summary['num_val_samples']}")
    print(f"  训练场景: {summary['num_train_scenes']}")
    print(f"  验证场景: {summary['num_val_scenes']}")
    print(f"  Ranking 组: {len(train_ranking_groups)} (训练), {len(val_ranking_groups)} (验证)")
    print("\n标签分布 (训练集):")
    for label, count in summary['train_label_distribution'].items():
        print(f"    {label}: {count}")
    print("="*60 + "\n")
    
    print("✅ 索引构建完成！")
    print(f"\n索引文件位置: {output_dir}")
    print("下一步: python train.py --config configs/train_consistency_mini.py")


if __name__ == '__main__':
    main()
