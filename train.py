#!/usr/bin/env python3
import argparse
import datetime
import importlib.util
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NuPlan critic training")
    parser.add_argument("--config", required=True, help="Python config path")
    parser.add_argument("--work-dir", type=str, default=None, help="Override work dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override workers")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Debug: cap train iterations per epoch")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Debug: cap val iterations per epoch")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("nuplan_critic_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "cfg"):
        raise ValueError(f"Config file must define `cfg`: {path}")
    cfg = dict(module.cfg)
    cfg["_config_path"] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_enabled() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> Dict[str, int]:
    if not is_dist_enabled():
        return {"rank": 0, "world_size": 1, "local_rank": 0}

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # 多节点训练需要较长的超时时间，避免因网络波动导致进程被误杀
    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# 全局标志：是否收到终止信号
_SIGTERM_RECEIVED = False


def _sigterm_handler(signum: int, frame: Any) -> None:
    """捕获 SIGTERM 信号，设置标志位让训练循环优雅退出"""
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True
    if is_main_process():
        print(
            "\n[WARNING] 收到 SIGTERM 信号，将在当前 step 结束后保存 checkpoint 并退出..."
        )


def sigterm_received() -> bool:
    """检查是否收到终止信号"""
    return _SIGTERM_RECEIVED


class CriticJsonlDataset(Dataset):
    def __init__(self, index_path: str, cfg: Dict[str, Any], training: bool) -> None:
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index file not found: {self.index_path}. "
                "Please build critic_train.jsonl / critic_val.jsonl first."
            )
        self.training = training
        self.image_root = Path(cfg["image_root"])
        self.image_size = int(cfg["image_size"])
        self.history_num_frames = int(cfg["history_num_frames"])
        self.candidate_traj_steps = int(cfg["candidate_traj_steps"])
        self.ego_state_dim = int(cfg["ego_state_dim"])
        self.traj_dim = int(cfg["traj_dim"])
        dataset_cfg = cfg.get("dataset", {})
        self.normalize_ego_state = bool(dataset_cfg.get("normalize_ego_state", True))
        self.normalize_candidate_traj = bool(dataset_cfg.get("normalize_candidate_traj", True))
        self.image_mean = torch.tensor(dataset_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
        self.image_std = torch.tensor(dataset_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=torch.float32)
        self.samples = self._load_jsonl()

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                required_keys = {"history_images", "ego_state", "candidate_traj", "label"}
                missing = required_keys - set(sample)
                if missing:
                    raise ValueError(
                        f"Missing keys {sorted(missing)} in {self.index_path}:{line_idx}"
                    )
                samples.append(sample)
        if not samples:
            raise ValueError(f"No samples found in {self.index_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        return path if path.is_absolute() else self.image_root / path

    def _load_image(self, image_path: str) -> torch.Tensor:
        path = self._resolve_image_path(image_path)
        with Image.open(path) as img:
            image = img.convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self.image_mean[:, None, None]) / self.image_std[:, None, None]
        return tensor

    def _prepare_history_images(self, image_paths: List[str]) -> torch.Tensor:
        selected = list(image_paths[-self.history_num_frames :])
        if len(selected) < self.history_num_frames:
            selected = [selected[0]] * (self.history_num_frames - len(selected)) + selected
        frames = [self._load_image(path) for path in selected]
        return torch.stack(frames, dim=0)

    def _prepare_vector(self, values: List[Any], length: int) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() < length:
            tensor = F.pad(tensor, (0, length - tensor.numel()))
        elif tensor.numel() > length:
            tensor = tensor[:length]
        return tensor

    def _prepare_traj(self, traj: List[List[Any]]) -> torch.Tensor:
        tensor = torch.tensor(traj, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"candidate_traj must be 2D, got shape={tuple(tensor.shape)}")
        steps, dims = tensor.shape
        if dims < self.traj_dim:
            tensor = F.pad(tensor, (0, self.traj_dim - dims))
        elif dims > self.traj_dim:
            tensor = tensor[:, : self.traj_dim]
        if steps < self.candidate_traj_steps:
            tensor = F.pad(tensor, (0, 0, 0, self.candidate_traj_steps - steps))
        elif steps > self.candidate_traj_steps:
            tensor = tensor[: self.candidate_traj_steps]
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        images = self._prepare_history_images(sample["history_images"])
        ego_state = self._prepare_vector(sample["ego_state"], self.ego_state_dim)
        candidate_traj = self._prepare_traj(sample["candidate_traj"])

        if self.normalize_ego_state:
            ego_state = torch.tanh(ego_state)
        if self.normalize_candidate_traj:
            candidate_traj = torch.tanh(candidate_traj)

        label = torch.tensor(float(sample["label"]), dtype=torch.float32)
        return {
            "images": images,
            "ego_state": ego_state,
            "candidate_traj": candidate_traj,
            "label": label,
        }


class SimpleImageEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, history, channels, height, width = images.shape
        x = images.reshape(batch_size * history, channels, height, width)
        x = self.backbone(x).flatten(1)
        x = self.proj(x)
        x = x.reshape(batch_size, history, -1).mean(dim=1)
        return x


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        return self.net(traj.flatten(1))


class CriticModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        image_dim = int(model_cfg["image_feature_dim"])
        action_dim = int(model_cfg["action_feature_dim"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg.get("dropout", 0.0))
        ego_state_dim = int(cfg["ego_state_dim"])
        candidate_traj_steps = int(cfg["candidate_traj_steps"])
        traj_dim = int(cfg["traj_dim"])

        self.image_encoder = SimpleImageEncoder(image_dim)
        self.traj_encoder = TrajectoryEncoder(candidate_traj_steps * traj_dim, hidden_dim, action_dim)
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_state_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.ReLU(inplace=True),
        )
        fusion_dim = image_dim + action_dim + action_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, images: torch.Tensor, ego_state: torch.Tensor, candidate_traj: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(images)
        traj_feat = self.traj_encoder(candidate_traj)
        ego_feat = self.ego_encoder(ego_state)
        fused = torch.cat([image_feat, traj_feat, ego_feat], dim=-1)
        return self.head(fused).squeeze(-1)



# ────────────────── Consistency Critic ──────────────────


class ConsistencyDataset(Dataset):
    """Consistency Critic 数据集，包含历史+未来图像和双标签"""

    def __init__(
        self, index_path: str, cfg: Dict[str, Any], training: bool,
    ) -> None:
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"索引文件不存在: {self.index_path}. "
                "请先运行 tools/build_consistency_index.py 生成索引。"
            )
        self.training = training
        self.image_root = Path(cfg["image_root"])
        self.image_size = int(cfg["image_size"])
        self.history_num_frames = int(cfg["history_num_frames"])
        self.future_num_frames = int(cfg["future_num_frames"])
        self.candidate_traj_steps = int(cfg["candidate_traj_steps"])
        self.ego_state_dim = int(cfg["ego_state_dim"])
        self.traj_dim = int(cfg["traj_dim"])
        ds_cfg = cfg.get("dataset", {})
        self.normalize_ego = bool(ds_cfg.get("normalize_ego_state", True))
        self.normalize_traj = bool(
            ds_cfg.get("normalize_candidate_traj", True),
        )
        self.normalize_mode: str = ds_cfg.get("normalize_mode", "tanh")
        traj_scale_raw = ds_cfg.get("traj_scale", None)
        if self.normalize_mode == "linear" and traj_scale_raw is None:
            raise ValueError(
                "normalize_mode='linear' 时必须在 dataset 配置中提供 traj_scale"
            )
        self.traj_scale: torch.Tensor | None = (
            torch.tensor(traj_scale_raw, dtype=torch.float32)
            if traj_scale_raw is not None
            else None
        )
        self.image_mean = torch.tensor(
            ds_cfg.get("image_mean", [0.485, 0.456, 0.406]),
            dtype=torch.float32,
        )
        self.image_std = torch.tensor(
            ds_cfg.get("image_std", [0.229, 0.224, 0.225]),
            dtype=torch.float32,
        )
        self.samples = self._load_jsonl()

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        required = {
            "history_images", "future_images", "ego_state",
            "candidate_traj", "consistency_label", "validity_label",
        }
        with self.index_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                missing = required - set(sample)
                if missing:
                    raise ValueError(
                        f"缺少字段 {sorted(missing)}，"
                        f"位于 {self.index_path}:{line_idx}"
                    )
                samples.append(sample)
        if not samples:
            raise ValueError(f"索引文件为空: {self.index_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, image_path: str) -> Path:
        p = Path(image_path)
        return p if p.is_absolute() else self.image_root / p

    def _load_image(self, image_path: str) -> torch.Tensor:
        path = self._resolve_path(image_path)
        with Image.open(path) as img:
            image = img.convert("RGB").resize(
                (self.image_size, self.image_size),
            )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (
            (tensor - self.image_mean[:, None, None])
            / self.image_std[:, None, None]
        )
        return tensor

    def _prepare_images(
        self, paths: List[str], num_frames: int,
    ) -> torch.Tensor:
        selected = list(paths[-num_frames:])
        if len(selected) < num_frames:
            selected = (
                [selected[0]] * (num_frames - len(selected)) + selected
            )
        return torch.stack([self._load_image(p) for p in selected], dim=0)

    def _prepare_vector(
        self, values: List[Any], length: int,
    ) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() < length:
            tensor = F.pad(tensor, (0, length - tensor.numel()))
        elif tensor.numel() > length:
            tensor = tensor[:length]
        return tensor

    def _prepare_traj(self, traj: List[List[Any]]) -> torch.Tensor:
        tensor = torch.tensor(traj, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(
                f"candidate_traj 必须为 2D，当前 shape={tuple(tensor.shape)}"
            )
        steps, dims = tensor.shape
        if dims < self.traj_dim:
            tensor = F.pad(tensor, (0, self.traj_dim - dims))
        elif dims > self.traj_dim:
            tensor = tensor[:, : self.traj_dim]
        if steps < self.candidate_traj_steps:
            tensor = F.pad(
                tensor, (0, 0, 0, self.candidate_traj_steps - steps),
            )
        elif steps > self.candidate_traj_steps:
            tensor = tensor[: self.candidate_traj_steps]
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        hist_imgs = self._prepare_images(
            sample["history_images"], self.history_num_frames,
        )
        fut_imgs = self._prepare_images(
            sample["future_images"], self.future_num_frames,
        )
        ego = self._prepare_vector(sample["ego_state"], self.ego_state_dim)
        traj = self._prepare_traj(sample["candidate_traj"])

        if self.normalize_ego:
            ego = torch.tanh(ego)
        if self.normalize_traj:
            if self.normalize_mode == "linear" and self.traj_scale is not None:
                traj = traj / self.traj_scale  # 广播 (steps, dim) / (dim,)
            else:
                traj = torch.tanh(traj)

        c_label = torch.tensor(
            float(sample["consistency_label"]), dtype=torch.float32,
        )
        v_label = torch.tensor(
            float(sample["validity_label"]), dtype=torch.float32,
        )
        return {
            "history_images": hist_imgs,
            "future_images": fut_imgs,
            "ego_state": ego,
            "candidate_traj": traj,
            "consistency_label": c_label,
            "validity_label": v_label,
        }


class ConsistencyCriticModel(nn.Module):
    """多维度 Action-Image Consistency Critic

    三层评估框架:
        Layer 1: 生成质量评估 (history + future image coherence)
        Layer 2: Action一致性评估 (speed, steering, progress, temporal)
        Layer 3: 驾驶合理性评估 (validity)
    
    结构:
        HistoryImageEncoder -> z_hist (256)
        FutureImageEncoder  -> z_future (256)
        TrajectoryEncoder   -> z_traj (128)
        EgoEncoder          -> z_ego (128)
        Concat -> z_all (768) -> SharedFusion (256)
        
        Heads:
        -> ConsistencyHead -> 1 (overall consistency)
        -> SpeedConsistencyHead -> 1 (speed consistency)
        -> SteeringConsistencyHead -> 1 (steering consistency)
        -> ProgressConsistencyHead -> 1 (progress consistency)
        -> TemporalCoherenceHead -> 1 (temporal coherence)
        -> ValidityHead -> 1 (driving validity)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        img_dim = int(mcfg["image_feature_dim"])
        act_dim = int(mcfg["action_feature_dim"])
        hidden = int(mcfg["hidden_dim"])
        fusion_dim = int(mcfg.get("fusion_dim", 256))
        dropout = float(mcfg.get("dropout", 0.0))
        ego_dim = int(cfg["ego_state_dim"])
        traj_steps = int(cfg["candidate_traj_steps"])
        traj_d = int(cfg["traj_dim"])

        # 共享 CNN backbone
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.history_proj = nn.Linear(256, img_dim)
        self.future_proj = nn.Linear(256, img_dim)

        self.traj_encoder = TrajectoryEncoder(
            traj_steps * traj_d, hidden, act_dim,
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, act_dim),
            nn.ReLU(inplace=True),
        )

        total_dim = img_dim * 2 + act_dim * 2
        self.shared_fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # 多维度评估头
        self.consistency_head = nn.Linear(fusion_dim, 1)  # overall consistency
        self.speed_consistency_head = nn.Linear(fusion_dim, 1)  # speed consistency
        self.steering_consistency_head = nn.Linear(fusion_dim, 1)  # steering consistency
        self.progress_consistency_head = nn.Linear(fusion_dim, 1)  # progress consistency
        self.temporal_coherence_head = nn.Linear(fusion_dim, 1)  # temporal coherence
        self.validity_head = nn.Linear(fusion_dim, 1)  # driving validity

    def _encode_images(
        self, images: torch.Tensor, proj: nn.Linear,
    ) -> torch.Tensor:
        """编码 (B, T, C, H, W) 图像序列为 (B, dim)"""
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        x = self.shared_backbone(x).flatten(1)
        x = proj(x)
        x = x.reshape(b, t, -1).mean(dim=1)
        return x

    def forward(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_hist = self._encode_images(history_images, self.history_proj)
        z_fut = self._encode_images(future_images, self.future_proj)
        z_traj = self.traj_encoder(candidate_traj)
        z_ego = self.ego_encoder(ego_state)

        z_all = torch.cat([z_hist, z_fut, z_traj, z_ego], dim=-1)
        z_shared = self.shared_fusion(z_all)

        return {
            # Layer 2: Action一致性评估（多维度）
            "consistency_logit": self.consistency_head(z_shared).squeeze(-1),
            "speed_consistency_logit": self.speed_consistency_head(z_shared).squeeze(-1),
            "steering_consistency_logit": self.steering_consistency_head(z_shared).squeeze(-1),
            "progress_consistency_logit": self.progress_consistency_head(z_shared).squeeze(-1),
            "temporal_coherence_logit": self.temporal_coherence_head(z_shared).squeeze(-1),
            # Layer 3: 驾驶合理性评估
            "validity_logit": self.validity_head(z_shared).squeeze(-1),
        }


def run_consistency_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
    training: bool,
    max_steps: int = 0,
) -> Dict[str, float]:
    """Consistency Critic 的单 epoch 训练/验证 - 多维度评估"""
    model.train(training)

    # 多维度损失权重
    lambda_c = float(cfg.get("lambda_consistency", 1.0))
    lambda_v = float(cfg.get("lambda_validity", 0.5))
    lambda_speed = float(cfg.get("lambda_speed_consistency", 0.3))
    lambda_steering = float(cfg.get("lambda_steering_consistency", 0.3))
    lambda_progress = float(cfg.get("lambda_progress_consistency", 0.2))
    lambda_temporal = float(cfg.get("lambda_temporal_coherence", 0.2))
    
    # 正样本权重
    c_pw = torch.tensor(
        cfg.get("consistency_positive_weight", cfg["positive_weight"]),
        device=device,
    )
    v_pw = torch.tensor(
        cfg.get("validity_positive_weight", cfg["positive_weight"]),
        device=device,
    )
    
    # 多维度损失函数
    criterion_c = nn.BCEWithLogitsLoss(pos_weight=c_pw)
    criterion_v = nn.BCEWithLogitsLoss(pos_weight=v_pw)
    criterion_speed = nn.BCEWithLogitsLoss()
    criterion_steering = nn.BCEWithLogitsLoss()
    criterion_progress = nn.BCEWithLogitsLoss()
    criterion_temporal = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_c_loss = 0.0
    total_v_loss = 0.0
    total_speed_loss = 0.0
    total_steering_loss = 0.0
    total_progress_loss = 0.0
    total_temporal_loss = 0.0
    
    total_c_correct = 0.0
    total_v_correct = 0.0
    total_speed_correct = 0.0
    total_steering_correct = 0.0
    total_progress_correct = 0.0
    total_temporal_correct = 0.0
    
    total_samples = 0
    log_interval = int(cfg["log_interval"])

    if training and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(loader, start=1):
        h_imgs = batch["history_images"].to(device, non_blocking=True)
        f_imgs = batch["future_images"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        c_labels = batch["consistency_label"].to(device, non_blocking=True)
        v_labels = batch["validity_label"].to(device, non_blocking=True)
        
        # 多维度标签（如果存在）
        speed_labels = batch.get("speed_consistency_label", c_labels).to(device, non_blocking=True)
        steering_labels = batch.get("steering_consistency_label", c_labels).to(device, non_blocking=True)
        progress_labels = batch.get("progress_consistency_label", c_labels).to(device, non_blocking=True)
        temporal_labels = batch.get("temporal_coherence_label", c_labels).to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            out = model(h_imgs, f_imgs, ego, traj)
            
            # 多维度损失计算
            loss_c = criterion_c(out["consistency_logit"], c_labels)
            loss_v = criterion_v(out["validity_logit"], v_labels)
            loss_speed = criterion_speed(out["speed_consistency_logit"], speed_labels)
            loss_steering = criterion_steering(out["steering_consistency_logit"], steering_labels)
            loss_progress = criterion_progress(out["progress_consistency_logit"], progress_labels)
            loss_temporal = criterion_temporal(out["temporal_coherence_logit"], temporal_labels)
            
            # 加权组合
            loss = (lambda_c * loss_c + 
                   lambda_v * loss_v + 
                   lambda_speed * loss_speed +
                   lambda_steering * loss_steering +
                   lambda_progress * loss_progress +
                   lambda_temporal * loss_temporal)
            
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = c_labels.size(0)
        
        # 多维度准确率计算
        c_preds = (torch.sigmoid(out["consistency_logit"]) >= 0.5).float()
        v_preds = (torch.sigmoid(out["validity_logit"]) >= 0.5).float()
        speed_preds = (torch.sigmoid(out["speed_consistency_logit"]) >= 0.5).float()
        steering_preds = (torch.sigmoid(out["steering_consistency_logit"]) >= 0.5).float()
        progress_preds = (torch.sigmoid(out["progress_consistency_logit"]) >= 0.5).float()
        temporal_preds = (torch.sigmoid(out["temporal_coherence_logit"]) >= 0.5).float()

        total_loss += loss.detach().item() * bs
        total_c_loss += loss_c.detach().item() * bs
        total_v_loss += loss_v.detach().item() * bs
        total_speed_loss += loss_speed.detach().item() * bs
        total_steering_loss += loss_steering.detach().item() * bs
        total_progress_loss += loss_progress.detach().item() * bs
        total_temporal_loss += loss_temporal.detach().item() * bs
        
        total_c_correct += (c_preds == c_labels).float().sum().item()
        total_v_correct += (v_preds == v_labels).float().sum().item()
        total_speed_correct += (speed_preds == speed_labels).float().sum().item()
        total_steering_correct += (steering_preds == steering_labels).float().sum().item()
        total_progress_correct += (progress_preds == progress_labels).float().sum().item()
        total_temporal_correct += (temporal_preds == temporal_labels).float().sum().item()
        
        total_samples += bs

        if is_main_process() and training and step % log_interval == 0:
            print(
                f"[Train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.detach().item():.4f} "
                f"c_loss={loss_c.detach().item():.4f} "
                f"v_loss={loss_v.detach().item():.4f}"
            )
        if max_steps and step >= max_steps:
            break
        if sigterm_received():
            if is_main_process():
                print(f"[WARNING] SIGTERM 中断训练，已完成 step={step}/{len(loader)}")
            break

    metrics = torch.tensor(
        [
            total_loss, total_c_loss, total_v_loss,
            total_speed_loss, total_steering_loss, total_progress_loss, total_temporal_loss,
            total_c_correct, total_v_correct,
            total_speed_correct, total_steering_correct, total_progress_correct, total_temporal_correct,
            float(total_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    metrics = reduce_mean(metrics)
    n = max(float(metrics[13].item()), 1.0)
    return {
        "loss": float(metrics[0].item() / n),
        "c_loss": float(metrics[1].item() / n),
        "v_loss": float(metrics[2].item() / n),
        "speed_loss": float(metrics[3].item() / n),
        "steering_loss": float(metrics[4].item() / n),
        "progress_loss": float(metrics[5].item() / n),
        "temporal_loss": float(metrics[6].item() / n),
        "c_acc": float(metrics[7].item() / n),
        "v_acc": float(metrics[8].item() / n),
        "speed_acc": float(metrics[9].item() / n),
        "steering_acc": float(metrics[10].item() / n),
        "progress_acc": float(metrics[11].item() / n),
        "temporal_acc": float(metrics[12].item() / n),
    }


def build_dataloader(cfg: Dict[str, Any], index_path: str, training: bool) -> DataLoader:
    model_type = cfg.get("model_type", "critic")
    if model_type == "consistency":
        dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=training)
    else:
        dataset = CriticJsonlDataset(index_path=index_path, cfg=cfg, training=training)
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=training, drop_last=training)
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=sampler is None and training,
        sampler=sampler,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=training,
    )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
    training: bool,
    max_steps: int = 0,
) -> Dict[str, float]:
    model.train(training)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["positive_weight"], device=device))
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    log_interval = int(cfg["log_interval"])

    if training and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        ego_state = batch["ego_state"].to(device, non_blocking=True)
        candidate_traj = batch["candidate_traj"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images, ego_state, candidate_traj)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_loss += loss.detach().item() * labels.size(0)
        total_correct += (preds == labels).float().sum().item()
        total_samples += labels.size(0)

        if is_main_process() and training and step % log_interval == 0:
            print(
                f"[Train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.detach().item():.4f}"
            )
        if max_steps and step >= max_steps:
            break
        if sigterm_received():
            if is_main_process():
                print(f"[WARNING] SIGTERM 中断训练，已完成 step={step}/{len(loader)}")
            break

    metrics = torch.tensor(
        [total_loss, total_correct, float(total_samples)],
        dtype=torch.float64,
        device=device,
    )
    metrics = reduce_mean(metrics)
    denom = max(float(metrics[2].item()), 1.0)
    return {
        "loss": float(metrics[0].item() / denom),
        "acc": float(metrics[1].item() / denom),
    }


def save_checkpoint(
    work_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    best_val_loss: float,
    is_best: bool,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "best_val_loss": best_val_loss,
    }
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pth"
    torch.save(state, latest_path)
    if is_best:
        torch.save(state, checkpoint_dir / "best.pth")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    # 注册 SIGTERM 信号处理器，收到终止信号时优雅退出
    signal.signal(signal.SIGTERM, _sigterm_handler)

    dist_info = setup_distributed()
    set_seed(int(cfg["seed"]) + dist_info["rank"])

    device = torch.device(
        f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu"
    )
    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(work_dir / "config_snapshot.json")
    if is_main_process():
        with (work_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    train_loader = build_dataloader(cfg, cfg["train_index"], training=True)
    val_loader = build_dataloader(cfg, cfg["val_index"], training=False)

    model_type = cfg.get("model_type", "critic")
    if model_type == "consistency":
        model = ConsistencyCriticModel(cfg).to(device)
    else:
        model = CriticModel(cfg).to(device)
    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]] if torch.cuda.is_available() else None,
            output_device=dist_info["local_rank"] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    optimizer_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )

    best_val_loss = math.inf
    total_epochs = int(cfg["epochs"])
    start_time = time.time()

    if is_main_process():
        print("=" * 60)
        title = ("NuPlan Consistency Critic Training"
                 if model_type == "consistency"
                 else "NuPlan Critic Training")
        print(title)
        print(f"Config: {cfg['_config_path']}")
        print(f"Work dir: {work_dir}")
        print(f"Device: {device}")
        print(f"World size: {dist_info['world_size']}")
        if torch.cuda.is_available():
            mem_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            print(f"GPU memory: {mem_total:.1f} GB")
        print("=" * 60)

    try:
        for epoch in range(1, total_epochs + 1):
            epoch_fn = (run_consistency_epoch
                        if model_type == "consistency"
                        else run_one_epoch)
            train_metrics = epoch_fn(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                cfg=cfg,
                training=True,
                max_steps=args.max_train_steps or 0,
            )
            val_metrics = epoch_fn(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                cfg=cfg,
                training=False,
                max_steps=args.max_val_steps or 0,
            )

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]

            if is_main_process():
                if model_type == "consistency":
                    print(
                        f"[Epoch {epoch}/{total_epochs}] "
                        f"loss={train_metrics['loss']:.4f} "
                        f"c_acc={train_metrics['c_acc']:.4f} "
                        f"v_acc={train_metrics['v_acc']:.4f} "
                        f"speed_acc={train_metrics['speed_acc']:.4f} "
                        f"steering_acc={train_metrics['steering_acc']:.4f} "
                        f"progress_acc={train_metrics['progress_acc']:.4f} "
                        f"temporal_acc={train_metrics['temporal_acc']:.4f} "
                        f"val_loss={val_metrics['loss']:.4f} "
                        f"val_c_acc={val_metrics['c_acc']:.4f} "
                        f"val_v_acc={val_metrics['v_acc']:.4f}"
                    )
                else:
                    print(
                        f"[Epoch {epoch}/{total_epochs}] "
                        f"train_loss={train_metrics['loss']:.4f} "
                        f"train_acc={train_metrics['acc']:.4f} "
                        f"val_loss={val_metrics['loss']:.4f} "
                        f"val_acc={val_metrics['acc']:.4f}"
                    )
                if epoch % int(cfg["save_interval"]) == 0:
                    save_checkpoint(
                        work_dir=work_dir,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        best_val_loss=best_val_loss,
                        is_best=is_best,
                    )

            # 收到 SIGTERM 时保存当前进度并退出
            if sigterm_received():
                if is_main_process():
                    print(f"[WARNING] 收到终止信号，保存 epoch={epoch} 的 checkpoint...")
                    save_checkpoint(
                        work_dir=work_dir,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        best_val_loss=best_val_loss,
                        is_best=False,
                    )
                    print("[WARNING] checkpoint 已保存，训练提前退出")
                break
    except Exception as e:
        # 打印详细错误信息，包含 GPU 显存状态，便于定位 OOM 等问题
        rank = dist_info["rank"]
        print(f"\n[ERROR][rank={rank}] 训练异常: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            print(
                f"[ERROR][rank={rank}] GPU 显存: "
                f"allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB",
                flush=True,
            )
        # 异常退出前尝试保存 checkpoint
        if is_main_process():
            try:
                print("[ERROR] 尝试保存紧急 checkpoint...", flush=True)
                save_checkpoint(
                    work_dir=work_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    best_val_loss=best_val_loss,
                    is_best=False,
                )
                print(f"[ERROR] 紧急 checkpoint 已保存至 {work_dir}/checkpoints/", flush=True)
            except Exception:
                print("[ERROR] 紧急 checkpoint 保存失败", flush=True)
        cleanup_distributed()
        sys.exit(1)

    if is_main_process():
        elapsed = time.time() - start_time
        print("=" * 60)
        print("Training finished")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Elapsed seconds: {elapsed:.1f}")
        print("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
