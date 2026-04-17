#!/usr/bin/env python3
"""
LPIPS (Learned Perceptual Image Patch Similarity) Calculator

用于评估生成图像与真实图像的感知相似度
Layer 1: 生成质量评估（感知级）

参考:
- Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", 2018
- https://github.com/richzhang/PerceptualSimilarity

作者: Your Name
日期: 2026-04-09
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from pathlib import Path
import warnings


class LPIPSCalculator:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) 计算器
    
    评估生成图像与真实图像的感知相似度
    越低越好（表示感知上更相似）
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        net: str = 'alex',
        batch_size: int = 32,
    ):
        """
        Args:
            device: 'cuda' 或 'cpu'
            net: 骨干网络 ('alex', 'vgg', 'squeeze')
            batch_size: 批处理大小
        """
        self.device = device
        self.net = net
        self.batch_size = batch_size
        
        # 加载 LPIPS 模型
        self.model = self._load_lpips_model(net=net, device=device)
        
        print(f"✅ LPIPSCalculator 初始化完成")
        print(f"   - Device: {device}")
        print(f"   - Network: {net}")
        print(f"   - Batch size: {batch_size}")
    
    def _load_lpips_model(self, net: str, device: str) -> nn.Module:
        """
        加载 LPIPS 模型
        
        Args:
            net: 骨干网络
            device: 设备
        
        Returns:
            LPIPS 模型
        """
        try:
            import lpips
            model = lpips.LPIPS(net=net, verbose=False)
            model.eval()
            model.to(device)
            
            # 冻结参数
            for param in model.parameters():
                param.requires_grad = False
            
            return model
            
        except ImportError:
            warnings.warn("未找到 lpips 库，使用简化实现")
            return self._create_simplified_lpips(net, device)
    
    def _create_simplified_lpips(self, net: str, device: str) -> nn.Module:
        """
        创建简化版 LPIPS（使用预训练 VGG）
        """
        from torchvision.models import vgg16, VGG16_Weights
        
        # 加载预训练 VGG
        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights)
        
        # 提取特征层
        class VGGFeatures(nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 relu3_3 层的特征
                self.features = nn.Sequential(*list(vgg.features.children())[:16])
                self.features.eval()
                for param in self.features.parameters():
                    param.requires_grad = False
            
            def forward(self, x):
                return self.features(x)
        
        return VGGFeatures().to(device)
    
    def compute_pairwise(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算两张图像的 LPIPS 距离
        
        Args:
            img1: (C, H, W) 或 (B, C, H, W)，归一化到 [-1, 1]
            img2: (C, H, W) 或 (B, C, H, W)，归一化到 [-1, 1]
        
        Returns:
            distance: LPIPS 距离（标量或 (B,)）
        """
        # 确保输入格式
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # 归一化到 [-1, 1]
        if img1.min() >= 0 and img1.max() <= 1:
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
        
        # 计算 LPIPS
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                # 使用 lpips 库
                distance = self.model(img1.to(self.device), img2.to(self.device))
            else:
                # 使用简化实现
                feat1 = self.model(img1.to(self.device))
                feat2 = self.model(img2.to(self.device))
                distance = torch.mean((feat1 - feat2) ** 2, dim=[1, 2, 3])
        
        return distance.squeeze()
    
    def compute(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
    ) -> float:
        """
        计算生成图像与真实图像的平均 LPIPS
        
        Args:
            generated: (N, C, H, W) 或 (N, T, C, H, W)，归一化到 [0, 1]
            real: (N, C, H, W) 或 (N, T, C, H, W)，归一化到 [0, 1]
        
        Returns:
            lpips: 平均 LPIPS 距离
        """
        # 处理视频输入
        if generated.dim() == 5:
            # (N, T, C, H, W) → (N*T, C, H, W)
            generated = generated.reshape(-1, *generated.shape[2:])
            real = real.reshape(-1, *real.shape[2:])
        
        # 确保数量匹配
        assert generated.shape[0] == real.shape[0], \
            f"图像数量不匹配: {generated.shape[0]} vs {real.shape[0]}"
        
        # 归一化到 [-1, 1]
        if generated.min() >= 0 and generated.max() <= 1:
            generated = generated * 2 - 1
            real = real * 2 - 1
        
        # 批量计算
        dataset = TensorDataset(generated, real)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        distances = []
        for batch_gen, batch_real in dataloader:
            dist = self.compute_pairwise(batch_gen, batch_real)
            distances.append(dist.cpu())
        
        all_distances = torch.cat(distances)
        avg_lpips = all_distances.mean().item()
        
        return avg_lpips
    
    def compute_per_scene(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
    ) -> list:
        """
        计算每个场景的 LPIPS
        
        Args:
            generated: (N, C, H, W) 或 (N, T, C, H, W)
            real: (N, C, H, W) 或 (N, T, C, H, W)
        
        Returns:
            distances: 每个场景的 LPIPS 距离列表
        """
        if generated.dim() == 5:
            generated = generated.reshape(-1, *generated.shape[2:])
            real = real.reshape(-1, *real.shape[2:])
        
        if generated.min() >= 0 and generated.max() <= 1:
            generated = generated * 2 - 1
            real = real * 2 - 1
        
        distances = []
        for i in range(generated.shape[0]):
            dist = self.compute_pairwise(generated[i:i+1], real[i:i+1])
            distances.append(dist.item())
        
        return distances


# 使用示例
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True,
                       help='生成数据文件 (.pt)')
    parser.add_argument('--real', type=str, required=True,
                       help='真实数据文件 (.pt)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--net', type=str, default='alex',
                       choices=['alex', 'vgg', 'squeeze'])
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    gen_data = torch.load(args.generated)
    real_data = torch.load(args.real)
    
    # 尝试不同的键名
    if 'generated_futures' in gen_data:
        generated = gen_data['generated_futures']
    elif 'generated' in gen_data:
        generated = gen_data['generated']
    else:
        raise KeyError("找不到生成数据")
    
    if 'ground_truth_future' in real_data:
        real = real_data['ground_truth_future']
    elif 'real' in real_data:
        real = real_data['real']
    else:
        raise KeyError("找不到真实数据")
    
    print(f"  生成数据: {generated.shape}")
    print(f"  真实数据: {real.shape}")
    
    # 初始化
    lpips_calc = LPIPSCalculator(
        device=args.device,
        net=args.net,
    )
    
    # 计算 LPIPS
    print(f"\n计算 LPIPS...")
    lpips_score = lpips_calc.compute(generated, real)
    
    print(f"\n{'='*60}")
    print(f"✅ LPIPS = {lpips_score:.4f}")
    print(f"{'='*60}")
    print(f"\n解释:")
    print(f"  - LPIPS 越低表示感知相似度越高")
    print(f"  - LPIPS = 0 表示完全相同")
    print(f"  - 比像素级指标（如 MSE）更符合人类感知")
    print(f"{'='*60}\n")
