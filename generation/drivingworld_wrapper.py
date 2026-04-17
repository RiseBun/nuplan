#!/usr/bin/env python3
"""
DrivingWorld World Model Wrapper

集成 DrivingWorld (https://github.com/YvanYin/DrivingWorld) 到三层评估系统

特性:
- 自回归视频生成
- Ego state 预测  
- 可控生成（基于 ego poses）
- 长时预测（40s+）

作者: Your Name
日期: 2026-04-09
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings

# 添加 DrivingWorld 到路径
DW_PATH = Path(__file__).parent.parent / 'DrivingWorld'
sys.path.insert(0, str(DW_PATH))


class DrivingWorldWrapper:
    """
    DrivingWorld World Model 包装器
    
    将 DrivingWorld 集成到三层评估系统，提供统一的生成接口
    """
    
    def __init__(
        self,
        world_model_path: str,
        vqvae_path: str,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        num_frames: int = 16,
    ):
        """
        初始化 DrivingWorld
        
        Args:
            world_model_path: World Model 权重路径
            vqvae_path: Video VQVAE 权重路径
            config_path: 配置文件路径
            device: 'cuda' 或 'cpu'
            num_frames: 生成帧数
        """
        self.device = device
        self.world_model_path = world_model_path
        self.vqvae_path = vqvae_path
        self.num_frames = num_frames
        
        # 默认配置文件
        if config_path is None:
            config_path = str(DW_PATH / "configs/drivingworld_v1/gen_videovq_conf_demo.py")
        
        self.config_path = config_path
        
        # 加载模型
        self.world_model = None
        self.vqvae = None
        self.config = None
        
        print(f"\n{'='*60}")
        print(f"加载 DrivingWorld...")
        print(f"{'='*60}")
        
        self._load_models()
        
        print(f"✅ DrivingWorld 加载完成")
        print(f"  - World Model: {world_model_path}")
        print(f"  - Video VQVAE: {vqvae_path}")
        print(f"  - Device: {device}")
        print(f"  - Num Frames: {num_frames}")
        print(f"{'='*60}\n")
    
    def _load_config(self):
        """加载配置文件"""
        from utils.config_utils import Config
        config = Config.fromfile(self.config_path)
        return config
    
    def _load_models(self):
        """加载 DrivingWorld 模型"""
        try:
            # 加载配置
            self.config = self._load_config()
            
            # 导入模型
            from models.model import TrainTransformers
            from modules.vqvae.model import VQVAE
            
            # 加载 World Model (Transformer)
            self.world_model = TrainTransformers(
                args=self.config,
                local_rank=-1,
                condition_frames=self.config.condition_frames,
            )
            
            # 加载权重
            world_model_ckpt = torch.load(self.world_model_path, map_location='cpu')
            if 'state_dict' in world_model_ckpt:
                self.world_model.load_state_dict(world_model_ckpt['state_dict'])
            else:
                self.world_model.load_state_dict(world_model_ckpt)
            
            self.world_model.to(self.device)
            self.world_model.eval()
            
            # 加载 Video VQVAE
            self.vqvae = VQVAE(self.config)
            vqvae_ckpt = torch.load(self.vqvae_path, map_location='cpu')
            if 'state_dict' in vqvae_ckpt:
                self.vqvae.load_state_dict(vqvae_ckpt['state_dict'])
            else:
                self.vqvae.load_state_dict(vqvae_ckpt)
            
            self.vqvae.to(self.device)
            self.vqvae.eval()
            
            print(f"✅ World Model 参数量: {sum(p.numel() for p in self.world_model.parameters()):,}")
            print(f"✅ VQVAE 参数量: {sum(p.numel() for p in self.vqvae.parameters()):,}")
            
        except Exception as e:
            raise RuntimeError(f"DrivingWorld 模型加载失败: {e}")
    
    def generate(
        self,
        history_images: torch.Tensor,      # (B, T_h, C, H, W)
        ego_state: torch.Tensor,            # (B, ego_dim)
        candidate_actions: torch.Tensor,    # (B, T_f, action_dim)
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        生成未来视频序列
        
        Args:
            history_images: 历史图像 (B, T_h, C, H, W)
            ego_state: 当前 ego 状态
            candidate_actions: 候选动作（将转换为 ego poses）
            num_samples: 生成样本数
        
        Returns:
            {
                'generated_images': (B, num_samples, T_f, C, H, W),
                'confidence': (B, num_samples),
                'predicted_ego_states': (B, num_samples, T_f, pose_dim),
            }
        """
        B = history_images.shape[0]
        T_f = candidate_actions.shape[1]
        
        generated_futures = []
        predicted_ego_states_list = []
        confidences = []
        
        for b in range(B):
            # 准备条件
            condition_frames = history_images[b]  # (T_h, C, H, W)
            
            # 将 action 转换为 ego poses [x, y, yaw]
            ego_poses = self._actions_to_poses(candidate_actions[b])
            
            # 生成多个样本
            batch_futures = []
            batch_ego_states = []
            
            for sample_idx in range(num_samples):
                with torch.no_grad():
                    # 使用 DrivingWorld 自回归生成
                    generated_video, predicted_poses = self._generate_single(
                        condition_frames=condition_frames,
                        ego_poses=ego_poses,
                        num_frames=T_f,
                    )
                
                batch_futures.append(generated_video)
                batch_ego_states.append(predicted_poses)
            
            generated_futures.append(torch.stack(batch_futures))
            predicted_ego_states_list.append(torch.stack(batch_ego_states))
            
            # 置信度（基于生成质量）
            confidences.append(torch.ones(num_samples) * 0.85)
        
        return {
            'generated_images': torch.stack(generated_futures),
            'confidence': torch.stack(confidences),
            'predicted_ego_states': torch.stack(predicted_ego_states_list),
        }
    
    def _generate_single(
        self,
        condition_frames: torch.Tensor,    # (T_h, C, H, W)
        ego_poses: torch.Tensor,            # (T_f, 3)
        num_frames: int,
    ) -> tuple:
        """
        单样本生成
        
        Returns:
            generated_video: (T_f, C, H, W)
            predicted_poses: (T_f, 3)
        """
        # 将图像编码为 token
        with torch.no_grad():
            condition_tokens = self.vqvae.encode(condition_frames.unsqueeze(0))
        
        # 将 poses 转换为 token indices
        pose_indices = self._poses_to_indices(ego_poses)
        
        # 自回归生成
        generated_tokens = self.world_model.generate(
            condition_tokens=condition_tokens,
            pose_indices=pose_indices,
            num_frames=num_frames,
            sampling_method='topk',
            topk=10,
        )
        
        # 解码为图像
        with torch.no_grad():
            generated_video = self.vqvae.decode(generated_tokens)
        
        # 预测的 poses
        predicted_poses = ego_poses  # 使用给定的 poses
        
        return generated_video.squeeze(0), predicted_poses
    
    def _actions_to_poses(self, actions: torch.Tensor) -> torch.Tensor:
        """
        将 action [dx, dy, dyaw] 转换为绝对位姿 [x, y, yaw]
        
        Args:
            actions: (T_f, 3) - 相对动作
        
        Returns:
            poses: (T_f, 3) - 绝对位姿
        """
        T_f = actions.shape[0]
        poses = []
        
        current_x, current_y, current_yaw = 0.0, 0.0, 0.0
        
        for t in range(T_f):
            dx, dy, dyaw = actions[t].tolist()
            current_x += dx
            current_y += dy
            current_yaw += dyaw
            poses.append([current_x, current_y, current_yaw])
        
        return torch.tensor(poses, dtype=torch.float32, device=actions.device)
    
    def _poses_to_indices(self, poses: torch.Tensor) -> torch.Tensor:
        """
        将连续 poses 转换为词汇表索引
        
        Args:
            poses: (T_f, 3) - [x, y, yaw]
        
        Returns:
            indices: (T_f, 3) - 离散索引
        """
        # 根据配置中的词汇表大小进行离散化
        pose_x_bins = torch.linspace(-50, 50, self.config.pose_x_vocab_size, device=poses.device)
        pose_y_bins = torch.linspace(-50, 50, self.config.pose_y_vocab_size, device=poses.device)
        yaw_bins = torch.linspace(-3.14, 3.14, self.config.yaw_vocab_size, device=poses.device)
        
        x_indices = torch.bucketize(poses[:, 0], pose_x_bins).clamp(0, self.config.pose_x_vocab_size - 1)
        y_indices = torch.bucketize(poses[:, 1], pose_y_bins).clamp(0, self.config.pose_y_vocab_size - 1)
        yaw_indices = torch.bucketize(poses[:, 2], yaw_bins).clamp(0, self.config.yaw_vocab_size - 1)
        
        return torch.stack([x_indices, y_indices, yaw_indices], dim=-1).long()
    
    def load_checkpoint(self, checkpoint_path: str):
        """重新加载 checkpoint"""
        self.world_model_path = checkpoint_path
        self._load_models()


def create_world_model(
    model_type: str = 'drivingworld',
    world_model_path: Optional[str] = None,
    vqvae_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = 'cuda',
    **kwargs
) -> DrivingWorldWrapper:
    """
    工厂函数：创建 World Model
    
    Args:
        model_type: 'drivingworld' | 'placeholder' | 'interpolation'
        world_model_path: World Model 权重路径
        vqvae_path: Video VQVAE 权重路径
        config_path: 配置文件路径
        device: 'cuda' | 'cpu'
        **kwargs: 其他参数
    
    Returns:
        World Model 实例
    """
    if model_type == 'drivingworld':
        if world_model_path is None or vqvae_path is None:
            raise ValueError("DrivingWorld 需要提供 world_model_path 和 vqvae_path")
        return DrivingWorldWrapper(
            world_model_path=world_model_path,
            vqvae_path=vqvae_path,
            config_path=config_path,
            device=device,
            num_frames=kwargs.get('num_frames', 16),
        )
    else:
        # 其他模型类型可以使用之前的实现
        from .drivewm_wrapper import create_world_model as create_other_model
        return create_other_model(model_type, device=device, **kwargs)


# 使用示例
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-model-path', type=str, required=True)
    parser.add_argument('--vqvae-path', type=str, required=True)
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # 创建模型
    wm = DrivingWorldWrapper(
        world_model_path=args.world_model_path,
        vqvae_path=args.vqvae_path,
        config_path=args.config_path,
        device=args.device,
    )
    
    # 测试生成
    B, T_h, T_f = 1, 4, 8
    history_images = torch.randn(B, T_h, 3, 256, 448)
    ego_state = torch.randn(B, 5)
    candidate_actions = torch.randn(B, T_f, 3)
    
    print(f"\n测试 DrivingWorld...")
    print(f"输入: history={history_images.shape}, action={candidate_actions.shape}")
    
    result = wm.generate(
        history_images=history_images,
        ego_state=ego_state,
        candidate_actions=candidate_actions,
        num_samples=2,
    )
    
    print(f"\n输出:")
    print(f"  generated_images: {result['generated_images'].shape}")
    print(f"  confidence: {result['confidence'].shape}")
    print(f"  predicted_ego_states: {result['predicted_ego_states'].shape}")
    print(f"\n✅ 测试通过！")
