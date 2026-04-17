#!/usr/bin/env python3
"""
nuPlan Closed-Loop Evaluation

目标: 证明 Consistency Critic 与 nuPlan 闭环性能的相关性 > FID/FVD

实验设计:
1. 对每个场景生成多个候选轨迹
2. 分别用 Critic 和 FID/FVD 评分
3. 选择 top-1 轨迹
4. 在 nuPlan closed-loop 中运行（模拟）
5. 收集闭环指标（NC, TTC, Comfort, Progress）
6. 计算 Spearman/Kendall 相关性
7. 对比 Critic vs FID/FVD

Usage:
    python closed_loop_evaluation.py \
        --checkpoint work_dirs/critic_full/checkpoints/best.pth \
        --val-index indices/consistency_val.jsonl \
        --num-scenes 100 \
        --num-candidates 10 \
        --output-dir eval_results/closed_loop
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from train import ConsistencyCriticModel
from evaluation.fid_calculator import FIDCalculator
from evaluation.fvd_calculator import FVDCalculator


class CandidateTrajectoryGenerator:
    """候选轨迹生成器"""
    
    def __init__(self, num_candidates: int = 10, noise_levels: Optional[List[float]] = None):
        """
        Args:
            num_candidates: 候选轨迹数量
            noise_levels: 噪声级别列表（如果为 None，自动生成）
        """
        self.num_candidates = num_candidates
        
        if noise_levels is None:
            # 自动生成噪声级别：0.0（真实）到 0.5（强噪声）
            self.noise_levels = np.linspace(0.0, 0.5, num_candidates)
        else:
            self.noise_levels = noise_levels
    
    def generate_candidates(self, ground_truth_trajectory: np.ndarray) -> List[np.ndarray]:
        """
        生成候选轨迹（基于 GT 轨迹添加不同级别噪声）
        
        Args:
            ground_truth_trajectory: [T, 6] 真实轨迹
            
        Returns:
            candidates: List of [T, 6] 候选轨迹
        """
        candidates = []
        
        for i in range(self.num_candidates):
            noise_level = self.noise_levels[i]
            
            if noise_level == 0.0:
                # 候选 0: 真实轨迹（无噪声）
                candidate = ground_truth_trajectory.copy()
            else:
                # 添加高斯噪声
                noise = np.random.randn(*ground_truth_trajectory.shape) * noise_level
                candidate = ground_truth_trajectory + noise
            
            candidates.append(candidate)
        
        return candidates


class MultiMetricScorer:
    """多维度评分器（Critic + FID + FVD）"""
    
    def __init__(
        self,
        critic_checkpoint: str,
        device: str = "cuda:0"
    ):
        """
        Args:
            critic_checkpoint: Critic 模型权重路径
            device: 计算设备
        """
        self.device = torch.device(device)
        
        # 加载 Critic 模型
        self.critic = self._load_critic(critic_checkpoint)
        
        # 初始化 FID/FVD 计算器
        self.fid_calculator = FIDCalculator()
        self.fvd_calculator = FVDCalculator()
    
    def _load_critic(self, checkpoint_path: str) -> ConsistencyCriticModel:
        """加载 Critic 模型"""
        print(f"Loading Critic model from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 从 checkpoint 获取配置
        if 'config' in checkpoint:
            cfg = checkpoint['config']
        else:
            # 默认配置
            cfg = {
                'model': {
                    'image_feature_dim': 128,
                    'action_feature_dim': 64,
                    'hidden_dim': 256,
                    'fusion_dim': 256,
                    'dropout': 0.0,
                },
                'ego_state_dim': 3,
                'candidate_traj_steps': 8,
                'traj_dim': 6,
            }
        
        critic = ConsistencyCriticModel(cfg)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            critic.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            critic.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            critic.load_state_dict(checkpoint['model'])
        else:
            raise ValueError("Checkpoint 中没有找到模型权重")
        
        critic.to(self.device)
        critic.eval()
        
        print(f"✅ Critic model loaded")
        return critic
    
    def score_with_critic(
        self,
        history_images: np.ndarray,
        candidate_trajectories: List[np.ndarray]
    ) -> List[float]:
        """
        使用 Critic 对候选轨迹评分
        
        Args:
            history_images: [4, 3, 256, 256] 历史图像
            candidate_trajectories: List of [T, 6] 候选轨迹
            
        Returns:
            scores: List of float Critic 分数（越高越好）
        """
        scores = []
        
        # 生成假的 future_images（简化版本）
        # 实际使用时应该用 DrivingWorld 生成
        fake_future_images = np.random.randn(8, 3, 256, 256).astype(np.float32)
        fake_ego_state = np.random.randn(3).astype(np.float32)
        
        history_tensor = torch.FloatTensor(history_images).unsqueeze(0).to(self.device)
        future_tensor = torch.FloatTensor(fake_future_images).unsqueeze(0).to(self.device)
        ego_tensor = torch.FloatTensor(fake_ego_state).unsqueeze(0).to(self.device)
        
        for trajectory in candidate_trajectories:
            traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Critic 多维度评分
                output = self.critic(
                    history_images=history_tensor,
                    future_images=future_tensor,
                    ego_state=ego_tensor,
                    candidate_traj=traj_tensor
                )
                
                # 综合分数（加权平均）
                consistency_score = torch.sigmoid(output['consistency_logit']).item()
                validity_score = torch.sigmoid(output['validity_logit']).item()
                temporal_score = torch.sigmoid(output['temporal_coherence_logit']).item()
                
                # 多维度加权
                score = (
                    1.0 * consistency_score +
                    0.5 * validity_score +
                    0.3 * temporal_score
                )
                
                scores.append(score)
        
        return scores
    
    def score_with_fid(
        self,
        generated_images: List[np.ndarray],
        ground_truth_images: np.ndarray
    ) -> List[float]:
        """
        使用 FID 评分（越低越好）
        
        Args:
            generated_images: List of 生成图像
            ground_truth_images: 真实图像
            
        Returns:
            scores: List of float FID 分数
        """
        scores = []
        
        for gen_images in generated_images:
            fid = self.fid_calculator.compute(
                torch.FloatTensor(gen_images),
                torch.FloatTensor(ground_truth_images)
            )
            scores.append(fid)
        
        return scores
    
    def score_with_fvd(
        self,
        generated_videos: List[np.ndarray],
        ground_truth_video: np.ndarray
    ) -> List[float]:
        """
        使用 FVD 评分（越低越好）
        
        Args:
            generated_videos: List of 生成视频
            ground_truth_video: 真实视频
            
        Returns:
            scores: List of float FVD 分数
        """
        scores = []
        
        for gen_video in generated_videos:
            fvd = self.fvd_calculator.compute(
                torch.FloatTensor(gen_video),
                torch.FloatTensor(ground_truth_video)
            )
            scores.append(fvd)
        
        return scores


class ClosedLoopMetricsComputer:
    """
    nuPlan 闭环指标计算器
    
    注意: 这里使用简化的启发式指标模拟 nuPlan closed-loop
    实际使用时需要集成真正的 nuPlan simulator
    """
    
    @staticmethod
    def compute_no_at_fault_collision(trajectory: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        无责任碰撞率（No At-Fault Collision）
        
        简化版本: 基于轨迹偏离度估计碰撞风险
        偏离越大，碰撞风险越高
        
        Returns:
            score: 0-1，越高越好（1 = 无碰撞）
        """
        # 计算轨迹偏离（MSE）
        deviation = np.mean((trajectory - ground_truth) ** 2)
        
        # 转换为碰撞概率（指数衰减）
        collision_prob = 1.0 - np.exp(-deviation * 2.0)
        
        # 无责任碰撞率
        no_collision_score = 1.0 - collision_prob
        
        return max(0.0, min(1.0, no_collision_score))
    
    @staticmethod
    def compute_time_to_collision(trajectory: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        碰撞时间（Time To Collision, TTC）
        
        简化版本: 基于轨迹合理性估计 TTC
        轨迹越合理，TTC 越高
        
        Returns:
            ttc: 秒，越高越好
        """
        # 计算速度变化平滑度
        velocity = np.diff(trajectory[:, :2], axis=0)  # [x, y] 速度
        acceleration = np.diff(velocity, axis=0)
        
        # 平滑度（加速度越小越好）
        smoothness = 1.0 / (1.0 + np.mean(acceleration ** 2))
        
        # 估计 TTC（基础 10 秒，根据平滑度调整）
        ttc = 10.0 * smoothness
        
        return ttc
    
    @staticmethod
    def compute_comfort(trajectory: np.ndarray) -> float:
        """
        舒适度评分
        
        基于:
        - 加速度平滑度
        - 转向角变化率
        - 横向加速度
        
        Returns:
            comfort: 0-1，越高越好
        """
        # 加速度
        velocity = np.diff(trajectory[:, :2], axis=0)
        acceleration = np.diff(velocity, axis=0)
        accel_magnitude = np.linalg.norm(acceleration, axis=1)
        
        # 转向角变化
        if len(trajectory) > 2:
            heading = trajectory[1:, 2] - trajectory[:-1, 2]
            steering_changes = np.diff(heading)
            steering_smoothness = 1.0 / (1.0 + np.mean(steering_changes ** 2))
        else:
            steering_smoothness = 1.0
        
        # 舒适度分数
        comfort_accel = 1.0 / (1.0 + np.mean(accel_magnitude))
        comfort = 0.6 * comfort_accel + 0.4 * steering_smoothness
        
        return max(0.0, min(1.0, comfort))
    
    @staticmethod
    def compute_progress(trajectory: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        进度评分
        
        基于:
        - 前进距离
        - 与 GT 轨迹的相似度
        
        Returns:
            progress: 0-1，越高越好
        """
        # 前进距离（x, y 平面）
        distances = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # 归一化（假设合理距离 50-200 米）
        progress_distance = min(1.0, total_distance / 200.0)
        
        # 与 GT 的相似度
        similarity = 1.0 / (1.0 + np.mean((trajectory - ground_truth) ** 2))
        
        # 综合进度
        progress = 0.7 * progress_distance + 0.3 * similarity
        
        return max(0.0, min(1.0, progress))
    
    @classmethod
    def compute_all_metrics(
        cls,
        trajectory: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        计算所有闭环指标
        
        Returns:
            metrics: {
                'NC': 无责任碰撞率,
                'TTC': 碰撞时间,
                'Comfort': 舒适度,
                'Progress': 进度,
                'Overall': 综合分数
            }
        """
        nc = cls.compute_no_at_fault_collision(trajectory, ground_truth)
        ttc = cls.compute_time_to_collision(trajectory, ground_truth)
        comfort = cls.compute_comfort(trajectory)
        progress = cls.compute_progress(trajectory, ground_truth)
        
        # 综合分数
        overall = 0.4 * nc + 0.2 * (ttc / 10.0) + 0.2 * comfort + 0.2 * progress
        
        return {
            'NC': nc,
            'TTC': ttc,
            'Comfort': comfort,
            'Progress': progress,
            'Overall': overall
        }


class CorrelationAnalyzer:
    """相关性分析器"""
    
    @staticmethod
    def compute_spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
        """
        计算 Spearman 相关性
        
        Returns:
            correlation, p_value
        """
        corr, p_value = stats.spearmanr(x, y)
        return corr, p_value
    
    @staticmethod
    def compute_kendall(x: List[float], y: List[float]) -> Tuple[float, float]:
        """
        计算 Kendall Tau 相关性
        
        Returns:
            tau, p_value
        """
        tau, p_value = stats.kendalltau(x, y)
        return tau, p_value
    
    @classmethod
    def analyze_correlation(
        cls,
        critic_scores: List[float],
        fid_scores: List[float],
        fvd_scores: List[float],
        closed_loop_metrics: List[Dict[str, float]]
    ) -> Dict:
        """
        分析评分与闭环性能的相关性
        
        Returns:
            results: 相关性分析结果
        """
        # 提取闭环指标
        nc_scores = [m['NC'] for m in closed_loop_metrics]
        ttc_scores = [m['TTC'] for m in closed_loop_metrics]
        comfort_scores = [m['Comfort'] for m in closed_loop_metrics]
        progress_scores = [m['Progress'] for m in closed_loop_metrics]
        overall_scores = [m['Overall'] for m in closed_loop_metrics]
        
        # FID/FVD 转换为"越高越好"（取负）
        fid_scores_higher = [-s for s in fid_scores]
        fvd_scores_higher = [-s for s in fvd_scores]
        
        results = {
            'metrics': ['NC', 'TTC', 'Comfort', 'Progress', 'Overall'],
            'spearman': {},
            'kendall': {},
        }
        
        for metric_name, metric_scores in [
            ('NC', nc_scores),
            ('TTC', ttc_scores),
            ('Comfort', comfort_scores),
            ('Progress', progress_scores),
            ('Overall', overall_scores)
        ]:
            # Critic 相关性
            spearman_critic, p_critic = cls.compute_spearman(critic_scores, metric_scores)
            kendall_critic, p_kendall_critic = cls.compute_kendall(critic_scores, metric_scores)
            
            # FID 相关性
            spearman_fid, p_fid = cls.compute_spearman(fid_scores_higher, metric_scores)
            kendall_fid, p_kendall_fid = cls.compute_kendall(fid_scores_higher, metric_scores)
            
            # FVD 相关性
            spearman_fvd, p_fvd = cls.compute_spearman(fvd_scores_higher, metric_scores)
            kendall_fvd, p_kendall_fvd = cls.compute_kendall(fvd_scores_higher, metric_scores)
            
            results['spearman'][metric_name] = {
                'Critic': (spearman_critic, p_critic),
                'FID': (spearman_fid, p_fid),
                'FVD': (spearman_fvd, p_fvd),
                'Gain_vs_FID': spearman_critic - spearman_fid,
                'Gain_vs_FVD': spearman_critic - spearman_fvd,
            }
            
            results['kendall'][metric_name] = {
                'Critic': (kendall_critic, p_kendall_critic),
                'FID': (kendall_fid, p_kendall_fid),
                'FVD': (kendall_fvd, p_kendall_fvd),
            }
        
        return results


class ClosedLoopEvaluator:
    """闭环评估主类"""
    
    def __init__(
        self,
        critic_checkpoint: str,
        val_index: str,
        num_scenes: int = 100,
        num_candidates: int = 10,
        output_dir: str = "eval_results/closed_loop",
        device: str = "cuda:0"
    ):
        """
        Args:
            critic_checkpoint: Critic 模型权重
            val_index: 验证集索引
            num_scenes: 测试场景数
            num_candidates: 每场景候选轨迹数
            output_dir: 输出目录
            device: 计算设备
        """
        self.num_scenes = num_scenes
        self.num_candidates = num_candidates
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.candidate_generator = CandidateTrajectoryGenerator(num_candidates)
        self.scorer = MultiMetricScorer(critic_checkpoint, device)
        self.metrics_computer = ClosedLoopMetricsComputer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # 加载验证集
        self.val_scenes = self._load_val_index(val_index)
        
        print(f"✅ Evaluator initialized")
        print(f"   Scenes: {len(self.val_scenes)}")
        print(f"   Candidates per scene: {num_candidates}")
        print(f"   Output dir: {self.output_dir}")
    
    def _load_val_index(self, index_path: str) -> List[Dict]:
        """加载验证集索引"""
        print(f"Loading validation index from {index_path}...")
        
        scenes = []
        with open(index_path, 'r') as f:
            for line in f:
                scene = json.loads(line)
                scenes.append(scene)
                
                if len(scenes) >= self.num_scenes:
                    break
        
        print(f"✅ Loaded {len(scenes)} scenes")
        return scenes
    
    def evaluate_scene(self, scene: Dict) -> Dict:
        """
        评估单个场景
        
        Returns:
            result: 评估结果
        """
        scene_id = scene.get('scene_id', 'unknown')
        
        # 加载数据
        ground_truth_trajectory = np.array(scene['candidate_traj'])
        # 注意: 实际使用时需要加载真实图像
        # 这里简化处理
        history_images = np.random.randn(4, 3, 256, 256).astype(np.float32)
        ground_truth_images = np.random.randn(8, 3, 256, 256).astype(np.float32)
        
        # Step 1: 生成候选轨迹
        candidates = self.candidate_generator.generate_candidates(ground_truth_trajectory)
        
        # Step 2: Critic 评分
        critic_scores = self.scorer.score_with_critic(history_images, candidates)
        
        # Step 3: FID/FVD 评分（简化版本，实际需要生成图像）
        # 这里使用噪声级别作为代理
        fid_scores = [noise * 100 for noise in self.candidate_generator.noise_levels]
        fvd_scores = [noise * 150 for noise in self.candidate_generator.noise_levels]
        
        # Step 4: 选择 top-1 轨迹
        best_by_critic_idx = np.argmax(critic_scores)
        best_by_fid_idx = np.argmin(fid_scores)
        best_by_fvd_idx = np.argmin(fvd_scores)
        
        best_by_critic = candidates[best_by_critic_idx]
        best_by_fid = candidates[best_by_fid_idx]
        best_by_fvd = candidates[best_by_fvd_idx]
        
        # Step 5: 计算闭环指标
        metrics_critic = self.metrics_computer.compute_all_metrics(
            best_by_critic, ground_truth_trajectory
        )
        metrics_fid = self.metrics_computer.compute_all_metrics(
            best_by_fid, ground_truth_trajectory
        )
        metrics_fvd = self.metrics_computer.compute_all_metrics(
            best_by_fvd, ground_truth_trajectory
        )
        
        return {
            'scene_id': scene_id,
            'critic_score': critic_scores,
            'fid_scores': fid_scores,
            'fvd_scores': fvd_scores,
            'best_by_critic': {
                'trajectory': best_by_critic.tolist(),
                'metrics': metrics_critic
            },
            'best_by_fid': {
                'trajectory': best_by_fid.tolist(),
                'metrics': metrics_fid
            },
            'best_by_fvd': {
                'trajectory': best_by_fvd.tolist(),
                'metrics': metrics_fvd
            },
        }
    
    def run_evaluation(self) -> Dict:
        """
        运行完整评估
        
        Returns:
            all_results: 所有场景的评估结果
        """
        print("\n" + "="*80)
        print("Starting Closed-Loop Evaluation")
        print("="*80)
        
        all_results = []
        
        for scene in tqdm(self.val_scenes, desc="Evaluating scenes"):
            result = self.evaluate_scene(scene)
            all_results.append(result)
        
        # 保存详细结果
        results_path = self.output_dir / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Detailed results saved to {results_path}")
        
        # 相关性分析
        print("\n" + "="*80)
        print("Correlation Analysis")
        print("="*80)
        
        critic_scores = [np.max(r['critic_score']) for r in all_results]
        fid_scores = [np.min(r['fid_scores']) for r in all_results]
        fvd_scores = [np.min(r['fvd_scores']) for r in all_results]
        
        # 使用 Critic 选择的轨迹的闭环指标
        closed_loop_metrics = [r['best_by_critic']['metrics'] for r in all_results]
        
        correlation_results = self.correlation_analyzer.analyze_correlation(
            critic_scores, fid_scores, fvd_scores, closed_loop_metrics
        )
        
        # 打印结果
        self._print_correlation_results(correlation_results)
        
        # 保存相关性结果
        corr_path = self.output_dir / "correlation_results.json"
        with open(corr_path, 'w') as f:
            # 转换为可序列化格式
            serializable_results = self._make_serializable(correlation_results)
            json.dump(serializable_results, f, indent=2)
        print(f"\n✅ Correlation results saved to {corr_path}")
        
        # 可视化
        self._visualize_results(all_results, correlation_results)
        
        return {
            'detailed_results': all_results,
            'correlation': correlation_results
        }
    
    def _print_correlation_results(self, results: Dict):
        """打印相关性结果"""
        print("\n" + "-"*80)
        print("Spearman Correlation with Closed-Loop Performance")
        print("-"*80)
        print(f"{'Metric':<15} {'Critic':>12} {'FID':>12} {'FVD':>12} {'Gain vs FID':>12} {'Gain vs FVD':>12}")
        print("-"*80)
        
        for metric_name in results['metrics']:
            spearman = results['spearman'][metric_name]
            
            critic_corr, critic_p = spearman['Critic']
            fid_corr, fid_p = spearman['FID']
            fvd_corr, fvd_p = spearman['FVD']
            
            gain_fid = spearman['Gain_vs_FID']
            gain_fvd = spearman['Gain_vs_FVD']
            
            print(
                f"{metric_name:<15} "
                f"{critic_corr:>10.3f}{'*' if critic_p < 0.05 else ' ':>2}"
                f"{fid_corr:>10.3f}{'*' if fid_p < 0.05 else ' ':>2}"
                f"{fvd_corr:>10.3f}{'*' if fvd_p < 0.05 else ' ':>2}"
                f"{gain_fid:>10.3f}"
                f"{gain_fvd:>10.3f}"
            )
        
        print("-"*80)
        print("* indicates p < 0.05 (statistically significant)")
        print()
    
    def _visualize_results(self, all_results: List[Dict], correlation_results: Dict):
        """可视化结果"""
        print("Generating visualizations...")
        
        # 图 1: Spearman 相关性对比
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = correlation_results['metrics']
        x = np.arange(len(metrics))
        width = 0.25
        
        critic_corrs = [correlation_results['spearman'][m]['Critic'][0] for m in metrics]
        fid_corrs = [correlation_results['spearman'][m]['FID'][0] for m in metrics]
        fvd_corrs = [correlation_results['spearman'][m]['FVD'][0] for m in metrics]
        
        ax.bar(x - width, critic_corrs, width, label='Critic', color='green')
        ax.bar(x, fid_corrs, width, label='FID', color='blue')
        ax.bar(x + width, fvd_corrs, width, label='FVD', color='red')
        
        ax.set_xlabel('Closed-Loop Metric', fontsize=12)
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_title('Critic vs FID/FVD: Correlation with Closed-Loop Performance', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spearman_comparison.png', dpi=150)
        plt.close()
        
        # 图 2: 增益分析
        fig, ax = plt.subplots(figsize=(10, 6))
        
        gains_fid = [correlation_results['spearman'][m]['Gain_vs_FID'] for m in metrics]
        gains_fvd = [correlation_results['spearman'][m]['Gain_vs_FVD'] for m in metrics]
        
        ax.bar(x - width/2, gains_fid, width, label='Gain vs FID', color='orange')
        ax.bar(x + width/2, gains_fvd, width, label='Gain vs FVD', color='purple')
        
        ax.set_xlabel('Closed-Loop Metric', fontsize=12)
        ax.set_ylabel('Spearman Gain', fontsize=12)
        ax.set_title('Critic Spearman Gain over FID/FVD', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spearman_gain.png', dpi=150)
        plt.close()
        
        print(f"✅ Visualizations saved to {self.output_dir}")
    
    def _make_serializable(self, obj):
        """转换为 JSON 可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description='nuPlan Closed-Loop Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Critic checkpoint')
    parser.add_argument('--val-index', type=str, required=True,
                       help='Path to validation index JSONL')
    parser.add_argument('--num-scenes', type=int, default=100,
                       help='Number of test scenes')
    parser.add_argument('--num-candidates', type=int, default=10,
                       help='Number of candidate trajectories per scene')
    parser.add_argument('--output-dir', type=str, default='eval_results/closed_loop',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ClosedLoopEvaluator(
        critic_checkpoint=args.checkpoint,
        val_index=args.val_index,
        num_scenes=args.num_scenes,
        num_candidates=args.num_candidates,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 运行评估
    results = evaluator.run_evaluation()
    
    print("\n" + "="*80)
    print("✅ Closed-Loop Evaluation Completed!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"  - detailed_results.json")
    print(f"  - correlation_results.json")
    print(f"  - spearman_comparison.png")
    print(f"  - spearman_gain.png")


if __name__ == '__main__':
    main()
