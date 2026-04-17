#!/usr/bin/env python3
"""
构建 Consistency Critic 训练索引

从标注好的数据构建 critic_train.jsonl 和 critic_val.jsonl

用法:
    python build_critic_index.py \
        --data-dir /mnt/cpfs/prediction/lipeinan/nuplan_data/critic_training_data_labeled \
        --output-dir /mnt/cpfs/prediction/lipeinan/nuplan/indices \
        --train-ratio 0.8
"""

import argparse
import torch
import json
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import random


def load_sample(sample_file: Path) -> Dict:
    """加载样本并转换为 JSON 格式"""
    sample = torch.load(sample_file)
    
    # 构建 JSON 格式
    json_sample = {
        # 图像路径（相对路径）
        "history_images": sample.get('image_paths', [])[:4],  # 前 4 帧作为历史
        "future_images": [],  # 这里使用生成的图像
        
        # Ego state 和轨迹
        "ego_state": sample['ego_state'].tolist(),
        "candidate_traj": sample['trajectory'].tolist(),
        
        # 多维度标签
        "consistency_label": sample['consistency_label'],
        "validity_label": sample['validity_label'],
        "speed_consistency_label": sample['speed_consistency_label'],
        "steering_consistency_label": sample['steering_consistency_label'],
        "progress_consistency_label": sample['progress_consistency_label'],
        "temporal_coherence_label": sample['temporal_coherence_label'],
        
        # 分数（用于分析）
        "fid_score": sample.get('fid_score', 0.0),
        "lpips_score": sample.get('lpips_score', 0.0),
        
        # 元数据
        "scene_name": sample['scene_name'],
        "camera_dir": sample['camera_dir'],
        "sample_type": sample.get('type', 'unknown'),
        "original_file": sample_file.name,
    }
    
    return json_sample


def main():
    parser = argparse.ArgumentParser(description='构建训练索引')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='标注数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/cpfs/prediction/lipeinan/nuplan/indices',
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--balance-classes', action='store_true',
                       help='是否平衡正负样本')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"构建训练索引")
    print(f"{'='*60}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  训练集比例: {args.train_ratio}")
    print(f"  平衡类别: {args.balance_classes}")
    print(f"{'='*60}\n")
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 查找所有样本文件
    data_dir = Path(args.data_dir)
    sample_files = sorted(data_dir.glob('*.pt'))
    
    if len(sample_files) == 0:
        print(f"❌ 未找到样本文件: {data_dir}")
        sys.exit(1)
    
    print(f"找到 {len(sample_files)} 个样本文件\n")
    
    # 加载所有样本
    print("加载样本...")
    samples = []
    for sample_file in tqdm(sample_files, desc="加载"):
        try:
            json_sample = load_sample(sample_file)
            samples.append(json_sample)
        except Exception as e:
            print(f"\n❌ 加载失败 {sample_file.name}: {e}")
            continue
    
    print(f"✅ 成功加载 {len(samples)} 个样本\n")
    
    # 平衡正负样本（可选）
    if args.balance_classes:
        print("平衡正负样本...")
        positive_samples = [s for s in samples if s['consistency_label'] == 1]
        negative_samples = [s for s in samples if s['consistency_label'] == 0]
        
        print(f"  正样本: {len(positive_samples)}")
        print(f"  负样本: {len(negative_samples)}")
        
        # 欠采样多数类
        if len(positive_samples) < len(negative_samples):
            negative_samples = random.sample(negative_samples, len(positive_samples))
        else:
            positive_samples = random.sample(positive_samples, len(negative_samples))
        
        samples = positive_samples + negative_samples
        random.shuffle(samples)
        
        print(f"  平衡后总数: {len(samples)}\n")
    
    # 划分训练集和验证集
    print("划分训练集和验证集...")
    random.shuffle(samples)
    
    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本\n")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为 JSONL 格式
    train_file = output_dir / 'consistency_train.jsonl'
    val_file = output_dir / 'consistency_val.jsonl'
    
    print(f"保存训练索引: {train_file}")
    with train_file.open('w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"保存验证索引: {val_file}")
    with val_file.open('w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 统计信息
    print(f"\n{'='*60}")
    print(f"索引构建完成！")
    print(f"{'='*60}")
    
    # 训练集统计
    train_pos = sum(1 for s in train_samples if s['consistency_label'] == 1)
    train_neg = sum(1 for s in train_samples if s['consistency_label'] == 0)
    
    print(f"\n训练集统计:")
    print(f"  总样本: {len(train_samples)}")
    print(f"  正样本: {train_pos} ({train_pos/len(train_samples)*100:.1f}%)")
    print(f"  负样本: {train_neg} ({train_neg/len(train_samples)*100:.1f}%)")
    
    # 验证集统计
    val_pos = sum(1 for s in val_samples if s['consistency_label'] == 1)
    val_neg = sum(1 for s in val_samples if s['consistency_label'] == 0)
    
    print(f"\n验证集统计:")
    print(f"  总样本: {len(val_samples)}")
    print(f"  正样本: {val_pos} ({val_pos/len(val_samples)*100:.1f}%)")
    print(f"  负样本: {val_neg} ({val_neg/len(val_samples)*100:.1f}%)")
    
    # 多维度标签统计
    print(f"\n多维度标签分布（训练集）:")
    for label_key in ['speed_consistency_label', 'steering_consistency_label',
                     'progress_consistency_label', 'temporal_coherence_label',
                     'validity_label']:
        pos_count = sum(1 for s in train_samples if s.get(label_key, 0) == 1)
        print(f"  {label_key}: {pos_count}/{len(train_samples)} ({pos_count/len(train_samples)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"下一步: 训练 Consistency Critic")
    print(f"{'='*60}")
    print(f"\n运行命令:")
    print(f"  python train.py \\")
    print(f"    --config configs/train_consistency_full.py \\")
    print(f"    --epochs 50 \\")
    print(f"    --batch-size 16 \\")
    print(f"    --work-dir work_dirs/critic_full")
    print(f"{'='*60}\n")
    
    # 保存构建元数据
    metadata = {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'train_positive': train_pos,
        'train_negative': train_neg,
        'val_positive': val_pos,
        'val_negative': val_neg,
        'balanced': args.balance_classes,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
    }
    
    metadata_file = output_dir / 'index_metadata.json'
    with metadata_file.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
