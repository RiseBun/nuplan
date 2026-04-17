#!/usr/bin/env python3
"""
生成 Consistency Critic 训练数据

使用 DrivingWorld 或其他 World Model 生成未来图像，构建多维度评估训练集。

功能:
1. 加载 nuPlan 场景的历史数据
2. 使用 World Model 生成多个候选未来
3. 保存 (history, generated_future, ego_state, action) 对

Usage:
    # 使用占位符生成（快速验证）
    python generate_futures_drivingworld.py \
        --output-dir generated_data \
        --num-scenes 100

    # 使用 DrivingWorld 生成
    python generate_futures_drivingworld.py \
        --output-dir generated_data \
        --world-model drivingworld \
        --world-model-path /path/to/world_model.pth \
        --vqvae-path /path/to/vqvae.pth \
        --num-scenes 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from generation.drivewm_wrapper import create_world_model, PlaceholderWorldModel
from generation.drivingworld_wrapper import DrivingWorldWrapper


@dataclass
class GenerationConfig:
    """生成配置"""
    # 数据源
    train_index: str = str(PROJECT_ROOT / "indices/critic_train.jsonl")
    val_index: str = str(PROJECT_ROOT / "indices/critic_val.jsonl")
    image_root: str = "/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set"
    
    # 生成参数
    world_model_type: str = "placeholder"  # placeholder | interpolation | drivingworld
    world_model_path: Optional[str] = None
    vqvae_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # 候选生成数量
    num_positive_samples: int = 1  # GT 轨迹作为正样本
    num_negative_samples: int = 4  # 扰动轨迹作为负样本
    
    # 扰动参数
    perturb_lateral_range: Tuple[float, float] = (0.5, 2.0)  # 横向扰动范围 (米)
    perturb_heading_range: Tuple[float, float] = (5.0, 15.0)  # 航向扰动范围 (度)
    perturb_speed_range: Tuple[float, float] = (0.7, 1.3)  # 速度扰动倍数
    
    # 生成设置
    num_frames: int = 4  # 未来帧数（用于生成）
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 输出
    output_dir: str = "generated_data"
    max_scenes: int = 0  # 0 表示全部


class NuPlanDataset(Dataset):
    """nuPlan 数据集加载器"""
    
    def __init__(self, index_path: str, image_root: str, max_scenes: int = 0):
        self.index_path = Path(index_path)
        self.image_root = Path(image_root)
        self.max_scenes = max_scenes
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载样本"""
        samples = []
        scene_groups = {}  # 按场景分组
        
        with open(self.index_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                scene_name = sample.get('scene_name', '')
                
                if scene_name not in scene_groups:
                    scene_groups[scene_name] = []
                scene_groups[scene_name].append(sample)
        
        # 限制场景数
        scene_names = sorted(scene_groups.keys())
        if self.max_scenes > 0:
            scene_names = scene_names[:self.max_scenes]
        
        # 收集锚点帧
        for scene_name in scene_names:
            for sample in scene_groups[scene_name]:
                # 只保留 GT 正样本
                if sample.get('label') == 1 and sample.get('sample_type') == 'gt_pos':
                    samples.append(sample)
                    break  # 每个场景只取一个锚点
        
        print(f"加载 {len(samples)} 个锚点样本 from {len(scene_names)} 场景")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class TrajectoryPerturber:
    """轨迹扰动器 - 生成负样本"""
    
    def __init__(
        self,
        lateral_range: Tuple[float, float] = (0.5, 2.0),
        heading_range: Tuple[float, float] = (5.0, 15.0),
        speed_range: Tuple[float, float] = (0.7, 1.3),
    ):
        self.lateral_range = lateral_range
        self.heading_range = heading_range
        self.speed_range = speed_range
    
    def perturb_trajectory(
        self,
        trajectory: np.ndarray,  # (T, 3) - [dx, dy, dyaw]
        perturb_type: str = 'random',
    ) -> np.ndarray:
        """
        扰动轨迹
        
        Args:
            trajectory: 原始轨迹 (T, 3)
            perturb_type: 'lateral' | 'heading' | 'speed' | 'random'
        
        Returns:
            扰动后的轨迹 (T, 3)
        """
        perturbed = trajectory.copy()
        T = len(trajectory)
        
        if perturb_type == 'lateral':
            # 横向偏移
            lateral_offset = np.random.uniform(*self.lateral_range)
            direction = np.random.choice([-1, 1])
            perturbed[:, 1] += direction * lateral_offset
            
        elif perturb_type == 'heading':
            # 航向扰动
            heading_offset = np.random.uniform(*self.heading_range) * np.pi / 180
            direction = np.random.choice([-1, 1])
            # 应用累计航向偏移
            cumulative_yaw = np.cumsum(perturbed[:, 2])
            perturbed[:, 2] += direction * heading_offset
            
        elif perturb_type == 'speed':
            # 速度缩放
            speed_scale = np.random.uniform(*self.speed_range)
            perturbed[:, 0] *= speed_scale
            perturbed[:, 1] *= speed_scale
            
        elif perturb_type == 'random':
            # 随机选择扰动类型
            perturb_type = np.random.choice(['lateral', 'heading', 'speed'])
            return self.perturb_trajectory(trajectory, perturb_type)
        
        return perturbed


class FutureGenerator:
    """未来图像生成器"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.perturber = TrajectoryPerturber(
            lateral_range=config.perturb_lateral_range,
            heading_range=config.perturb_heading_range,
            speed_range=config.speed_range,
        )
        self.world_model = self._create_world_model()
    
    def _create_world_model(self):
        """创建 World Model"""
        print(f"\n初始化 World Model: {self.config.world_model_type}")
        
        if self.config.world_model_type == 'drivingworld':
            if self.config.world_model_path is None or self.config.vqvae_path is None:
                raise ValueError("DrivingWorld 需要提供 world_model_path 和 vqvae_path")
            
            return DrivingWorldWrapper(
                world_model_path=self.config.world_model_path,
                vqvae_path=self.config.vqvae_path,
                config_path=self.config.config_path,
                device=self.config.device,
                num_frames=self.config.num_frames,
            )
        else:
            return create_world_model(
                model_type=self.config.world_model_type,
                device=self.config.device,
            )
    
    def generate_samples(
        self,
        anchor_sample: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        为一个锚点样本生成多个候选未来
        
        Returns:
            List of {
                'history_images': [...],
                'future_images': [...],  # 生成的未来图像路径
                'ego_state': [...],
                'candidate_traj': [...],
                'label': 0 or 1,
                'sample_type': str,
            }
        """
        # 解析锚点数据
        history_image_paths = anchor_sample['history_images']
        ego_state = anchor_sample['ego_state']
        gt_traj = np.array(anchor_sample['candidate_traj'])
        
        # 加载历史图像
        history_images = self._load_history_images(history_image_paths)
        
        # 生成多个候选
        results = []
        
        # 1. GT 正样本（使用 GT 轨迹生成）
        gt_sample = self._create_sample(
            history_images=history_images,
            ego_state=ego_state,
            trajectory=gt_traj,
            label=1,
            sample_type='gt_pos',
        )
        results.append(gt_sample)
        
        # 2. GT 负样本（使用扰动轨迹生成）
        perturb_types = ['lateral', 'heading', 'speed']
        for i, perturb_type in enumerate(perturb_types):
            perturbed_traj = self.perturber.perturb_trajectory(gt_traj, perturb_type)
            sample = self._create_sample(
                history_images=history_images,
                ego_state=ego_state,
                trajectory=perturbed_traj,
                label=0,
                sample_type=f'perturb_{perturb_type}',
            )
            results.append(sample)
        
        return results
    
    def _load_history_images(self, image_paths: List[str]) -> torch.Tensor:
        """加载历史图像序列"""
        from PIL import Image
        
        images = []
        for path in image_paths:
            full_path = Path(self.config.image_root) / path if not Path(path).is_absolute() else Path(path)
            with Image.open(full_path) as img:
                img = img.convert('RGB').resize((256, 256))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
                images.append(tensor)
        
        return torch.stack(images)  # (T_h, C, H, W)
    
    def _create_sample(
        self,
        history_images: torch.Tensor,
        ego_state: List[float],
        trajectory: np.ndarray,
        label: int,
        sample_type: str,
    ) -> Dict[str, Any]:
        """创建样本（包含生成逻辑）"""
        B, T_h, C, H, W = history_images.shape
        
        # 转换为 Tensor
        ego_state_tensor = torch.tensor(ego_state, dtype=torch.float32).unsqueeze(0)
        traj_tensor = torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0)
        
        # 生成未来图像
        T_f = self.config.num_frames
        
        # 确保 trajectory 有正确的帧数
        if len(trajectory) > T_f:
            traj_tensor = traj_tensor[:, :T_f]
        elif len(trajectory) < T_f:
            pad_len = T_f - len(trajectory)
            padding = torch.zeros(1, pad_len, trajectory.shape[1])
            traj_tensor = torch.cat([traj_tensor, padding], dim=1)
        
        # 移动到设备
        history_images = history_images.unsqueeze(0).to(self.config.device)
        ego_state_tensor = ego_state_tensor.to(self.config.device)
        traj_tensor = traj_tensor.to(self.config.device)
        
        # 生成
        with torch.no_grad():
            # 对于占位符模型，action 需要是正确的 shape
            generated = self.world_model.generate(
                history_images=history_images,
                ego_state=ego_state_tensor,
                candidate_actions=traj_tensor,
                num_samples=1,
            )
        
        # 简化：保存生成结果的元数据（不保存实际图像）
        # 实际应用中会保存图像到磁盘
        return {
            'history_images': history_images.squeeze(0).cpu().numpy().tolist(),  # 用于验证
            'ego_state': ego_state,
            'candidate_traj': trajectory.tolist(),
            'label': label,
            'sample_type': sample_type,
            'generation_metadata': {
                'world_model': self.config.world_model_type,
                'num_frames': T_f,
            },
        }


def main():
    parser = argparse.ArgumentParser(description='生成 Consistency Critic 训练数据')
    
    # 数据源
    parser.add_argument('--train-index', type=str, default=None,
                       help='训练索引文件')
    parser.add_argument('--val-index', type=str, default=None,
                       help='验证索引文件')
    parser.add_argument('--image-root', type=str, default=None,
                       help='图像根目录')
    
    # 生成参数
    parser.add_argument('--world-model', type=str, default='placeholder',
                       choices=['placeholder', 'interpolation', 'drivingworld'],
                       help='World Model 类型')
    parser.add_argument('--world-model-path', type=str, default=None,
                       help='World Model 权重路径')
    parser.add_argument('--vqvae-path', type=str, default=None,
                       help='VQVAE 权重路径')
    parser.add_argument('--config-path', type=str, default=None,
                       help='配置文件路径')
    
    # 候选数量
    parser.add_argument('--num-positive', type=int, default=1,
                       help='正样本数量')
    parser.add_argument('--num-negative', type=int, default=4,
                       help='负样本数量')
    
    # 输出
    parser.add_argument('--output-dir', type=str, default='generated_data',
                       help='输出目录')
    parser.add_argument('--max-scenes', type=int, default=0,
                       help='最大场景数，0表示全部')
    
    # 其他
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='设备')
    
    args = parser.parse_args()
    
    # 加载默认配置
    config = GenerationConfig()
    
    # 覆盖配置
    if args.train_index:
        config.train_index = args.train_index
    if args.val_index:
        config.val_index = args.val_index
    if args.image_root:
        config.image_root = args.image_root
    if args.world_model:
        config.world_model_type = args.world_model
    if args.world_model_path:
        config.world_model_path = args.world_model_path
    if args.vqvae_path:
        config.vqvae_path = args.vqvae_path
    if args.config_path:
        config.config_path = args.config_path
    if args.num_positive:
        config.num_positive_samples = args.num_positive
    if args.num_negative:
        config.num_negative_samples = args.num_negative
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_scenes:
        config.max_scenes = args.max_scenes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # 输出配置
    print("\n" + "="*60)
    print("生成配置")
    print("="*60)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建生成器
    generator = FutureGenerator(config)
    
    # 处理训练集
    print("\n处理训练集...")
    train_dataset = NuPlanDataset(
        index_path=config.train_index,
        image_root=config.image_root,
        max_scenes=config.max_scenes,
    )
    
    train_samples = []
    for i, anchor in enumerate(tqdm(train_dataset, desc="生成训练数据")):
        try:
            generated = generator.generate_samples(anchor)
            train_samples.extend(generated)
        except Exception as e:
            print(f"\n处理样本 {i} 失败: {e}")
            continue
        
        # 进度报告
        if (i + 1) % 100 == 0:
            print(f"\n进度: {i+1}/{len(train_dataset)}, 已生成 {len(train_samples)} 样本")
    
    # 保存训练数据
    train_output_file = output_dir / "generated_train.jsonl"
    with open(train_output_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\n保存训练数据: {train_output_file} ({len(train_samples)} 样本)")
    
    # 处理验证集
    print("\n处理验证集...")
    val_dataset = NuPlanDataset(
        index_path=config.val_index,
        image_root=config.image_root,
        max_scenes=max(10, config.max_scenes // 5) if config.max_scenes > 0 else 50,
    )
    
    val_samples = []
    for i, anchor in enumerate(tqdm(val_dataset, desc="生成验证数据")):
        try:
            generated = generator.generate_samples(anchor)
            val_samples.extend(generated)
        except Exception as e:
            print(f"\n处理样本 {i} 失败: {e}")
            continue
    
    # 保存验证数据
    val_output_file = output_dir / "generated_val.jsonl"
    with open(val_output_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\n保存验证数据: {val_output_file} ({len(val_samples)} 样本)")
    
    # 保存生成统计
    stats = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                   for k, v in vars(config).items()},
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'train_by_type': {},
        'val_by_type': {},
    }
    
    # 统计样本类型
    for sample in train_samples:
        sample_type = sample.get('sample_type', 'unknown')
        stats['train_by_type'][sample_type] = stats['train_by_type'].get(sample_type, 0) + 1
    for sample in val_samples:
        sample_type = sample.get('sample_type', 'unknown')
        stats['val_by_type'][sample_type] = stats['val_by_type'].get(sample_type, 0) + 1
    
    stats_file = output_dir / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("生成完成")
    print("="*60)
    print(f"  训练样本: {len(train_samples)}")
    print(f"  验证样本: {len(val_samples)}")
    print(f"  统计文件: {stats_file}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
