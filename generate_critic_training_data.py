#!/usr/bin/env python3
"""
生成 Consistency Critic 训练数据

使用 DrivingWorld 生成未来图像，构建正负样本对

数据路径:
- 输入: /mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set
- 输出: /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data

用法:
    python generate_critic_training_data.py \
        --data-root /mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set \
        --output-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data \
        --num-scenes 500 \
        --samples-per-scene 5 \
        --device cuda
"""

import argparse
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'DrivingWorld'))

from generation.drivingworld_wrapper import DrivingWorldWrapper


class NuPlanSceneLoader:
    """nuPlan 场景加载器"""
    
    def __init__(
        self,
        data_root: str,
        history_frames: int = 4,
        future_frames: int = 8,
    ):
        self.data_root = Path(data_root)
        self.history_frames = history_frames
        self.future_frames = future_frames
        
        # 发现场景
        self.scenes = self._discover_scenes()
        print(f"✅ 找到 {len(self.scenes)} 个场景")
    
    def _discover_scenes(self) -> List[Dict]:
        """发现所有可用场景"""
        scenes = []
        
        # 遍历相机目录
        for camera_dir in self.data_root.glob('nuplan-v1.1_mini_camera_*'):
            if not camera_dir.is_dir():
                continue
            
            for scene_dir in camera_dir.glob('*'):
                if not scene_dir.is_dir():
                    continue
                
                cam_f0 = scene_dir / 'CAM_F0'
                if cam_f0.exists():
                    images = sorted(list(cam_f0.glob('*.jpg')))
                    
                    # 需要足够的帧
                    min_frames = self.history_frames + self.future_frames
                    if len(images) >= min_frames:
                        scenes.append({
                            'scene_name': scene_dir.name,
                            'camera_dir': camera_dir.name,
                            'image_dir': cam_f0,
                            'images': images,
                        })
        
        return scenes
    
    def load_scene(self, scene_idx: int) -> Dict:
        """
        加载场景数据
        
        Returns:
            {
                'scene_name': str,
                'history_images': (T_h, C, H, W),
                'future_images': (T_f, C, H, W),
                'ego_state': (5,),
                'trajectory': (T_f, 3),
            }
        """
        scene = self.scenes[scene_idx]
        
        # 加载图像
        transform = transforms.Compose([
            transforms.Resize((256, 448)),  # DrivingWorld 分辨率
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        images = []
        for img_path in scene['images'][:self.history_frames + self.future_frames]:
            with Image.open(img_path) as img:
                img_tensor = transform(img.convert('RGB'))
                images.append(img_tensor)
        
        images_tensor = torch.stack(images)
        
        # 分割历史和未来
        history_images = images_tensor[:self.history_frames]
        future_images = images_tensor[self.history_frames:]
        
        # 模拟 ego state 和 trajectory（实际应该从 nuPlan DB 读取）
        # 这里使用简化的估计
        ego_state = torch.randn(5)  # [vx, vy, yaw, ax, angular_rate]
        
        # 从图像估计简单轨迹（实际应该用真实轨迹）
        trajectory = self._estimate_trajectory(future_images)
        
        return {
            'scene_name': scene['scene_name'],
            'camera_dir': scene['camera_dir'],
            'history_images': history_images,
            'future_images': future_images,
            'ego_state': ego_state,
            'trajectory': trajectory,
            'image_paths': [str(p) for p in scene['images'][:self.history_frames + self.future_frames]],
        }
    
    def _estimate_trajectory(self, future_images: torch.Tensor) -> torch.Tensor:
        """
        从未来图像估计简单轨迹
        
        这是一个简化版本，实际应该使用 nuPlan 的真实轨迹数据
        """
        T_f = future_images.shape[0]
        
        # 简化的轨迹估计（基于光流或特征点跟踪）
        # 这里使用随机轨迹作为占位符
        trajectory = torch.randn(T_f, 3) * 0.5  # [dx, dy, dyaw]
        
        return trajectory
    
    def __len__(self):
        return len(self.scenes)


class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(
        self,
        world_model_path: str,
        vqvae_path: str,
        device: str = 'cuda',
    ):
        self.device = device
        
        # 加载 World Model
        print("\n加载 DrivingWorld...")
        self.world_model = DrivingWorldWrapper(
            world_model_path=world_model_path,
            vqvae_path=vqvae_path,
            device=device,
            num_frames=8,
        )
    
    def generate_samples(
        self,
        scene_data: Dict,
        num_samples: int = 5,
    ) -> List[Dict]:
        """
        为单个场景生成多个训练样本
        
        Args:
            scene_data: 场景数据
            num_samples: 生成样本数（1 正样本 + num_samples-1 负样本）
        
        Returns:
            samples: 训练样本列表
        """
        history = scene_data['history_images'].unsqueeze(0).to(self.device)
        ego_state = scene_data['ego_state'].unsqueeze(0).to(self.device)
        ground_truth_future = scene_data['future_images']
        real_trajectory = scene_data['trajectory']
        
        samples = []
        
        # 样本 0: 正样本（使用真实轨迹）
        print("  生成正样本...")
        with torch.no_grad():
            positive_result = self.world_model.generate(
                history_images=history,
                ego_state=ego_state,
                candidate_actions=real_trajectory.unsqueeze(0),
                num_samples=1,
            )
        
        samples.append({
            'type': 'positive',
            'history_images': scene_data['history_images'],
            'generated_future': positive_result['generated_images'][0, 0],
            'ground_truth_future': ground_truth_future,
            'trajectory': real_trajectory,
            'ego_state': scene_data['ego_state'],
            'scene_name': scene_data['scene_name'],
            'camera_dir': scene_data['camera_dir'],
            'label': 1,  # 正样本
        })
        
        # 样本 1~N-1: 负样本（使用扰动的轨迹）
        print(f"  生成 {num_samples - 1} 个负样本...")
        for i in range(num_samples - 1):
            # 扰动轨迹
            noise_level = 0.5 + i * 0.3  # 逐渐增加噪声
            noisy_trajectory = real_trajectory + torch.randn_like(real_trajectory) * noise_level
            
            with torch.no_grad():
                negative_result = self.world_model.generate(
                    history_images=history,
                    ego_state=ego_state,
                    candidate_actions=noisy_trajectory.unsqueeze(0),
                    num_samples=1,
                )
            
            samples.append({
                'type': 'negative',
                'history_images': scene_data['history_images'],
                'generated_future': negative_result['generated_images'][0, 0],
                'ground_truth_future': ground_truth_future,
                'trajectory': noisy_trajectory,
                'ego_state': scene_data['ego_state'],
                'scene_name': scene_data['scene_name'],
                'camera_dir': scene_data['camera_dir'],
                'label': 0,  # 负样本
                'noise_level': noise_level,
            })
        
        return samples


def main():
    parser = argparse.ArgumentParser(description='生成 Consistency Critic 训练数据')
    parser.add_argument('--data-root', type=str,
                       default='/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set',
                       help='nuPlan 数据根目录')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data',
                       help='输出目录')
    parser.add_argument('--world-model-path', type=str,
                       default='DrivingWorld/pretrained_models/world_model.pth')
    parser.add_argument('--vqvae-path', type=str,
                       default='DrivingWorld/pretrained_models/video_vqvae.pth')
    parser.add_argument('--num-scenes', type=int, default=500,
                       help='场景数量')
    parser.add_argument('--samples-per-scene', type=int, default=5,
                       help='每个场景的样本数（1 正 + N-1 负）')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--history-frames', type=int, default=4)
    parser.add_argument('--future-frames', type=int, default=8)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Consistency Critic 训练数据生成")
    print(f"{'='*60}")
    print(f"  数据目录: {args.data_root}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  场景数量: {args.num_scenes}")
    print(f"  每场景样本: {args.samples_per_scene}")
    print(f"{'='*60}\n")
    
    # 初始化场景加载器
    scene_loader = NuPlanSceneLoader(
        data_root=args.data_root,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
    )
    
    # 限制场景数
    if len(scene_loader) > args.num_scenes:
        print(f"限制到前 {args.num_scenes} 个场景")
        scene_loader.scenes = scene_loader.scenes[:args.num_scenes]
    
    # 初始化生成器
    generator = TrainingDataGenerator(
        world_model_path=args.world_model_path,
        vqvae_path=args.vqvae_path,
        device=args.device,
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成数据
    print(f"\n开始生成训练数据...")
    print(f"  总场景数: {len(scene_loader)}")
    print(f"  总样本数: {len(scene_loader) * args.samples_per_scene}\n")
    
    metadata = {
        'num_scenes': len(scene_loader),
        'samples_per_scene': args.samples_per_scene,
        'total_samples': 0,
        'positive_samples': 0,
        'negative_samples': 0,
        'generated_files': [],
    }
    
    for scene_idx in tqdm(range(len(scene_loader)), desc="生成场景"):
        try:
            # 加载场景
            scene_data = scene_loader.load_scene(scene_idx)
            
            # 生成样本
            samples = generator.generate_samples(
                scene_data=scene_data,
                num_samples=args.samples_per_scene,
            )
            
            # 保存样本
            for sample_idx, sample in enumerate(samples):
                output_file = output_dir / f"scene_{scene_idx:05d}_sample_{sample_idx}.pt"
                
                torch.save(sample, output_file)
                metadata['generated_files'].append(output_file.name)
                metadata['total_samples'] += 1
                
                if sample['label'] == 1:
                    metadata['positive_samples'] += 1
                else:
                    metadata['negative_samples'] += 1
            
        except Exception as e:
            print(f"\n❌ 场景 {scene_idx} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存元数据
    metadata_file = output_dir / 'generation_metadata.json'
    with metadata_file.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ 生成完成！")
    print(f"{'='*60}")
    print(f"  总场景数: {metadata['num_scenes']}")
    print(f"  总样本数: {metadata['total_samples']}")
    print(f"    正样本: {metadata['positive_samples']}")
    print(f"    负样本: {metadata['negative_samples']}")
    print(f"  输出目录: {output_dir}")
    print(f"  元数据: {metadata_file}")
    print(f"{'='*60}\n")
    
    print("下一步:")
    print(f"1. 计算训练标签:")
    print(f"   python compute_training_labels.py \\")
    print(f"     --data-dir {output_dir} \\")
    print(f"     --output-dir {output_dir}_labeled")
    print(f"\n2. 构建训练索引:")
    print(f"   python build_critic_index.py \\")
    print(f"     --data-dir {output_dir}_labeled \\")
    print(f"     --output-dir indices")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
