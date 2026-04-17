#!/usr/bin/env python3
"""
FVD (Fréchet Video Distance) Calculator

用于评估生成视频与真实视频的时空分布距离
Layer 1: 生成质量评估（视频级）

参考:
- Unterthiner et al. "Towards Accurate Generative Models of Video: A New Metric & Challenges", 2018
- https://github.com/universome/fvd-comparison

作者: Your Name
日期: 2026-04-09
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from pathlib import Path
import warnings


class I3DFeatureExtractor(nn.Module):
    """
    I3D (Inflated 3D ConvNet) 特征提取器
    
    用于提取视频的时空特征
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: 'cuda' 或 'cpu'
        """
        super().__init__()
        self.device = device
        
        # 尝试加载预训练的 I3D
        try:
            from pytorch_i3d import InceptionI3d
            self.model = InceptionI3d(400, in_channels=3)
            
            # 下载预训练权重（首次运行）
            # 从 https://github.com/piergiaj/pytorch-i3d 下载
            self._load_pretrained_weights()
            
        except ImportError:
            # 如果没有 pytorch_i3d，使用简化的 3D ResNet
            warnings.warn("未找到 pytorch_i3d，使用 3D ResNet 作为替代")
            self.model = self._create_3d_resnet()
        
        self.model.eval()
        self.model.to(device)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_pretrained_weights(self):
        """加载 I3D 预训练权重"""
        # 这里需要手动下载权重
        # 或者使用 torchvision 的 video model
        pass
    
    def _create_3d_resnet(self):
        """创建 3D ResNet 作为替代"""
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            # 移除最后的分类层
            model.fc = nn.Identity()
            return model
        except ImportError:
            raise ImportError("需要安装 torchvision: pip install torchvision")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取视频特征
        
        Args:
            x: (B, T, C, H, W) 视频序列，归一化到 [0, 1]
        
        Returns:
            features: (B, feature_dim)
        """
        # 转换为 (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        # 调整大小到 I3D 需要的尺寸
        if x.shape[-2:] != (224, 224):
            x = nn.functional.interpolate(
                x.reshape(-1, *x.shape[2:]),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            ).reshape(x.shape[0], x.shape[1], x.shape[2], 224, 224)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(x)
        
        return features


class FVDCalculator:
    """
    FVD (Fréchet Video Distance) 计算器
    
    评估生成视频与真实视频的时空分布距离
    越低越好（表示生成分布更接近真实分布）
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 8,
    ):
        """
        Args:
            device: 'cuda' 或 'cpu'
            batch_size: 批处理大小（视频级）
        """
        self.device = device
        self.batch_size = batch_size
        self.feature_extractor = I3DFeatureExtractor(device=device)
        
        print(f"✅ FVDCalculator 初始化完成")
        print(f"   - Device: {device}")
        print(f"   - Batch size: {batch_size}")
    
    def compute_statistics(
        self,
        videos: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算视频特征的统计量 (mu, sigma)
        
        Args:
            videos: (N, T, C, H, W) 视频序列
        
        Returns:
            mu: (feature_dim,) 特征均值
            sigma: (feature_dim, feature_dim) 特征协方差矩阵
        """
        # 创建 DataLoader
        dataset = TensorDataset(videos)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # 提取特征
        features = []
        for (batch,) in dataloader:
            batch = batch.to(self.device)
            feat = self.feature_extractor(batch)
            features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        
        # 计算统计量
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def compute_fvd(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """
        计算 FVD 分数
        
        FVD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        
        Args:
            mu1, sigma1: 生成视频的特征统计量
            mu2, sigma2: 真实视频的特征统计量
        
        Returns:
            fvd: FVD 分数（越低越好）
        """
        diff = mu1 - mu2
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值误差
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # 计算 FVD
        fvd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fvd)
    
    def compute(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
    ) -> float:
        """
        计算生成视频与真实视频的 FVD
        
        Args:
            generated: (N, T, C, H, W) 生成视频，归一化到 [0, 1]
            real: (N, T, C, H, W) 真实视频，归一化到 [0, 1]
        
        Returns:
            fvd: FVD 分数
        """
        # 归一化到 [0, 1]
        if generated.min() < 0 or generated.max() > 1:
            generated = (generated + 1) / 2
            warnings.warn("Generated videos 已归一化到 [0, 1]")
        
        if real.min() < 0 or real.max() > 1:
            real = (real + 1) / 2
            warnings.warn("Real videos 已归一化到 [0, 1]")
        
        # 计算统计量
        print("  计算生成视频特征...")
        mu_gen, sigma_gen = self.compute_statistics(generated)
        
        print("  计算真实视频特征...")
        mu_real, sigma_real = self.compute_statistics(real)
        
        # 计算 FVD
        fvd = self.compute_fvd(mu_gen, sigma_gen, mu_real, sigma_real)
        
        return fvd


# 简化版 FVD（如果不使用 I3D）
class SimplifiedFVD:
    """
    简化版 FVD
    
    使用预训练的 2D 网络提取每帧特征，然后聚合
    不需要 I3D 模型，更容易使用
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 32,
    ):
        self.device = device
        self.batch_size = batch_size
        
        # 使用 InceptionV3 提取每帧特征
        from evaluation.fid_calculator import InceptionV3FeatureExtractor
        self.feature_extractor = InceptionV3FeatureExtractor(device=device)
    
    def compute(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
        aggregate: str = 'mean',
    ) -> float:
        """
        计算简化版 FVD
        
        Args:
            generated: (N, T, C, H, W) 生成视频
            real: (N, T, C, H, W) 真实视频
            aggregate: 聚合方式 ('mean', 'max', 'concat')
        
        Returns:
            fvd: 简化版 FVD 分数
        """
        N, T, C, H, W = generated.shape
        
        # 重塑为 (N*T, C, H, W)
        gen_flat = generated.reshape(-1, C, H, W)
        real_flat = real.reshape(-1, C, H, W)
        
        # 提取每帧特征
        gen_features = self.feature_extractor(gen_flat)  # (N*T, D)
        real_features = self.feature_extractor(real_flat)  # (N*T, D)
        
        # 聚合时序信息
        gen_features = gen_features.reshape(N, T, -1)
        real_features = real_features.reshape(N, T, -1)
        
        if aggregate == 'mean':
            gen_agg = gen_features.mean(dim=1)  # (N, D)
            real_agg = real_features.mean(dim=1)
        elif aggregate == 'max':
            gen_agg = gen_features.max(dim=1)[0]
            real_agg = real_features.max(dim=1)[0]
        else:
            gen_agg = gen_features.reshape(N, -1)
            real_agg = real_features.reshape(N, -1)
        
        # 计算统计量
        mu_gen = gen_agg.mean(dim=0).cpu().numpy()
        sigma_gen = np.cov(gen_agg.cpu().numpy(), rowvar=False)
        
        mu_real = real_agg.mean(dim=0).cpu().numpy()
        sigma_real = np.cov(real_agg.cpu().numpy(), rowvar=False)
        
        # 计算 FVD
        from scipy import linalg
        diff = mu_gen - mu_real
        covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fvd = diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_real) - 2 * np.trace(covmean)
        
        return float(fvd)


# 使用示例
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True,
                       help='生成视频文件 (.pt)')
    parser.add_argument('--real', type=str, required=True,
                       help='真实视频文件 (.pt)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--simplified', action='store_true',
                       help='使用简化版 FVD')
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    gen_data = torch.load(args.generated)
    real_data = torch.load(args.real)
    
    generated = gen_data['generated_futures']  # (N, T, C, H, W)
    real = real_data['ground_truth_future']
    
    print(f"  生成视频: {generated.shape}")
    print(f"  真实视频: {real.shape}")
    
    # 计算 FVD
    if args.simplified:
        print("\n使用简化版 FVD...")
        fvd_calc = SimplifiedFVD(device=args.device)
    else:
        print("\n使用完整版 FVD (I3D)...")
        fvd_calc = FVDCalculator(device=args.device)
    
    print(f"\n计算 FVD...")
    fvd = fvd_calc.compute(generated, real)
    
    print(f"\n{'='*60}")
    print(f"✅ FVD = {fvd:.2f}")
    print(f"{'='*60}")
    print(f"\n解释:")
    print(f"  - FVD 越低表示生成质量越好")
    print(f"  - FVD = 0 表示完全相同的分布")
    print(f"  - 考虑了时序一致性")
    print(f"{'='*60}\n")
