#!/usr/bin/env python3
"""
FID (Fréchet Inception Distance) Calculator

用于评估生成图像与真实图像的分布距离
Layer 1: 生成质量评估

参考:
- Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", 2017
- https://github.com/mseitzer/pytorch-fid

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

# 使用 torchvision 的 InceptionV3
try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    from torchvision import transforms
except ImportError:
    raise ImportError("需要安装 torchvision: pip install torchvision")


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 特征提取器"""
    
    def __init__(self, dims: int = 2048, device: str = 'cuda', offline_mode: bool = False):
        """
        Args:
            dims: 特征维度 (64, 192, 768, 2048)
            device: 'cuda' 或 'cpu'
            offline_mode: 离线模式，跳过预训练权重下载
        """
        super().__init__()
        self.device = device
        self.dims = dims
        
        # 加载 InceptionV3
        try:
            if offline_mode:
                # 离线模式：不加载预训练权重
                print("[INFO] 离线模式：使用随机初始化的 InceptionV3（仅用于调试）")
                self.model = inception_v3(weights=None, transform_input=False)
            else:
                # 在线模式：尝试下载预训练权重
                weights = Inception_V3_Weights.DEFAULT
                self.model = inception_v3(weights=weights, transform_input=False)
        except Exception as e:
            print(f"[WARNING] 无法加载预训练权重: {e}")
            print("[INFO] 使用随机初始化的 InceptionV3（仅用于调试）")
            self.model = inception_v3(weights=None, transform_input=False)
        
        self.model.eval()
        self.model.to(device)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        
        Args:
            x: (B, 3, 299, 299) 归一化到 [0, 1]
        
        Returns:
            features: (B, dims)
        """
        # InceptionV3 需要输入大小为 299x299
        if x.shape[2:] != (299, 299):
            x = nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # 提取特征
        with torch.no_grad():
            # 获取 pool3 层的特征 (2048 维)
            x = self.model.Conv2d_1a_3x3(x)
            x = self.model.Conv2d_2a_3x3(x)
            x = self.model.Conv2d_2b_3x3(x)
            x = self.model.maxpool1(x)
            x = self.model.Conv2d_3b_1x1(x)
            x = self.model.Conv2d_4a_3x3(x)
            x = self.model.maxpool2(x)
            x = self.model.Mixed_5b(x)
            x = self.model.Mixed_5c(x)
            x = self.model.Mixed_5d(x)
            x = self.model.Mixed_6a(x)
            x = self.model.Mixed_6b(x)
            x = self.model.Mixed_6c(x)
            x = self.model.Mixed_6d(x)
            x = self.model.Mixed_6e(x)
            x = self.model.Mixed_7a(x)
            x = self.model.Mixed_7b(x)
            x = self.model.Mixed_7c(x)
            
            # 自适应平均池化
            features = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            features = features.view(features.size(0), -1)
        
        return features


class FIDCalculator:
    """
    FID (Fréchet Inception Distance) 计算器
    
    评估生成图像分布与真实图像分布的距离
    越低越好（表示生成分布更接近真实分布）
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        dims: int = 2048,
        batch_size: int = 32,
        offline_mode: bool = False,
    ):
        """
        Args:
            device: 'cuda' 或 'cpu'
            dims: 特征维度
            batch_size: 批处理大小
            offline_mode: 离线模式，跳过预训练权重下载
        """
        self.device = device
        self.batch_size = batch_size
        self.feature_extractor = InceptionV3FeatureExtractor(
            dims=dims, 
            device=device,
            offline_mode=offline_mode
        )
        
        print(f"✅ FIDCalculator 初始化完成")
        print(f"   - Device: {device}")
        print(f"   - Feature dims: {dims}")
        print(f"   - Batch size: {batch_size}")
    
    def compute_statistics(
        self,
        images: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算图像特征的统计量 (mu, sigma)
        
        Args:
            images: (N, C, H, W) 归一化到 [0, 1]
        
        Returns:
            mu: (dims,) 特征均值
            sigma: (dims, dims) 特征协方差矩阵
        """
        # 创建 DataLoader
        dataset = TensorDataset(images)
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
    
    def compute_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """
        计算 FID 分数
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        
        Args:
            mu1, sigma1: 生成图像的特征统计量
            mu2, sigma2: 真实图像的特征统计量
        
        Returns:
            fid: FID 分数（越低越好）
        """
        diff = mu1 - mu2
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值误差
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # 计算 FID
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fid)
    
    def compute(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
    ) -> float:
        """
        计算生成图像与真实图像的 FID
        
        Args:
            generated: (N, C, H, W) 生成图像，归一化到 [0, 1]
            real: (N, C, H, W) 真实图像，归一化到 [0, 1]
        
        Returns:
            fid: FID 分数
        """
        # 确保输入格式正确
        if generated.dim() == 5:
            # (B, T, C, H, W) → (B*T, C, H, W)
            generated = generated.reshape(-1, *generated.shape[2:])
            real = real.reshape(-1, *real.shape[2:])
        
        # 归一化到 [0, 1]
        if generated.min() < 0 or generated.max() > 1:
            generated = (generated + 1) / 2  # 假设原来是 [-1, 1]
            warnings.warn("Generated images 已归一化到 [0, 1]")
        
        if real.min() < 0 or real.max() > 1:
            real = (real + 1) / 2
            warnings.warn("Real images 已归一化到 [0, 1]")
        
        # 计算统计量
        print("  计算生成图像特征...")
        mu_gen, sigma_gen = self.compute_statistics(generated)
        
        print("  计算真实图像特征...")
        mu_real, sigma_real = self.compute_statistics(real)
        
        # 计算 FID
        fid = self.compute_fid(mu_gen, sigma_gen, mu_real, sigma_real)
        
        return fid
    
    def compute_from_files(
        self,
        generated_dir: str,
        real_dir: str,
    ) -> float:
        """
        从目录加载图像并计算 FID
        
        Args:
            generated_dir: 生成图像目录
            real_dir: 真实图像目录
        
        Returns:
            fid: FID 分数
        """
        from PIL import Image
        
        # 加载图像
        def load_images_from_dir(dir_path):
            images = []
            for img_file in sorted(Path(dir_path).glob('*')):
                if img_file.suffix in ['.jpg', '.jpeg', '.png']:
                    img = Image.open(img_file).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((299, 299)),
                        transforms.ToTensor(),
                    ])
                    images.append(transform(img))
            return torch.stack(images)
        
        print(f"加载生成图像: {generated_dir}")
        generated = load_images_from_dir(generated_dir)
        
        print(f"加载真实图像: {real_dir}")
        real = load_images_from_dir(real_dir)
        
        return self.compute(generated, real)


# 使用示例
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True,
                       help='生成图像路径（文件或目录）')
    parser.add_argument('--real', type=str, required=True,
                       help='真实图像路径（文件或目录）')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    # 初始化
    fid_calc = FIDCalculator(
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # 计算 FID
    print(f"\n计算 FID...")
    print(f"  生成: {args.generated}")
    print(f"  真实: {args.real}\n")
    
    fid = fid_calc.compute_from_files(args.generated, args.real)
    
    print(f"\n{'='*60}")
    print(f"✅ FID = {fid:.2f}")
    print(f"{'='*60}")
    print(f"\n解释:")
    print(f"  - FID 越低表示生成质量越好")
    print(f"  - FID = 0 表示完全相同的分布")
    print(f"  - 好的模型通常 FID < 50")
    print(f"{'='*60}\n")
