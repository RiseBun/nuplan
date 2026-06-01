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
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NuPlan critic training")
    parser.add_argument("--config", required=True, help="Python config path")
    parser.add_argument("--work-dir", type=str, default=None, help="Override work dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override workers")
    parser.add_argument(
        "--baseline-mode",
        choices=["full", "no_image", "ego_only", "no_traj", "traj_only"],
        default=None,
        help="P0 shortcut audit baseline mode for consistency critic",
    )
    parser.add_argument("--max-train-steps", type=int, default=None, help="Debug: cap train iterations per epoch")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Debug: cap val iterations per epoch")
    parser.add_argument(
        "--preflight-samples",
        type=int,
        default=128,
        help="Validate image paths from each index before training; 0 disables, -1 checks all rows.",
    )
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
        self.consistency_traj_steps = int(
            cfg.get("consistency_traj_steps", min(self.future_num_frames, self.candidate_traj_steps)),
        )
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
        self.consistency_source_weights = {
            str(key): float(value)
            for key, value in cfg.get("consistency_source_weights", {}).items()
        }
        self.label_quality_weights = {
            str(key): float(value)
            for key, value in cfg.get("label_quality_weights", {}).items()
        }
        self.default_consistency_weight = float(
            cfg.get("default_consistency_weight", 1.0),
        )
        self.validity_negative_weight = float(
            cfg.get("validity_negative_weight", 1.0),
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

    def selected_image_paths(
        self, sample: Dict[str, Any], key: str, num_frames: int,
    ) -> List[Path]:
        paths = list(sample[key][-num_frames:])
        if not paths:
            raise ValueError(f"样本缺少图像路径字段: {key}")
        if len(paths) < num_frames:
            paths = [paths[0]] * (num_frames - len(paths)) + paths
        return [self._resolve_path(path) for path in paths]

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
        source_type = str(sample.get("source_type", sample.get("sample_type", "unknown")))
        label_quality = str(sample.get("label_quality", "clean_negative" if c_label.item() == 0.0 else "positive"))
        quality_weight = self.label_quality_weights.get(label_quality, 1.0)
        c_weight = torch.tensor(
            self.consistency_source_weights.get(
                source_type, self.default_consistency_weight,
            ) * quality_weight,
            dtype=torch.float32,
        )
        v_weight = torch.tensor(
            self.validity_negative_weight if float(v_label.item()) == 0.0 else 1.0,
            dtype=torch.float32,
        )
        return {
            "history_images": hist_imgs,
            "future_images": fut_imgs,
            "ego_state": ego,
            "candidate_traj": traj,
            "consistency_label": c_label,
            "validity_label": v_label,
            "consistency_weight": c_weight,
            "validity_weight": v_weight,
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


class ConsistencyCriticModel(nn.Module):
    """P0-audited Action-Image Consistency Critic

    结构:
        HistoryImageEncoder -> z_hist (256)
        FutureImageEncoder  -> z_future (256)
        ConsistencyTrajectoryEncoder -> z_traj_consistency (128)
        ValidityTrajectoryEncoder    -> z_traj_validity (128)
        EgoEncoder          -> z_ego (128)

    P0 约束:
        Consistency 只看与 future images 对齐的前 consistency_traj_steps 步轨迹。
        Validity 只看 ego + 完整轨迹，不接图像特征，避免场景 shortcut。
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
        consistency_traj_steps = int(
            cfg.get("consistency_traj_steps", min(int(cfg.get("future_num_frames", traj_steps)), traj_steps)),
        )
        traj_d = int(cfg["traj_dim"])
        self.baseline_mode = str(cfg.get("baseline_mode", "full"))
        self.consistency_traj_steps = consistency_traj_steps
        self.use_action_visual_interaction = bool(
            mcfg.get("use_action_visual_interaction", False),
        )
        self.temporal_encoder_type = str(mcfg.get("temporal_encoder", "mean"))
        if self.temporal_encoder_type not in {"mean", "gru"}:
            raise ValueError(
                "model.temporal_encoder must be one of: mean, gru",
            )

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
        if self.temporal_encoder_type == "gru":
            self.history_temporal_encoder = nn.GRU(
                input_size=img_dim,
                hidden_size=img_dim,
                batch_first=True,
            )
            self.future_temporal_encoder = nn.GRU(
                input_size=img_dim,
                hidden_size=img_dim,
                batch_first=True,
            )
        else:
            self.history_temporal_encoder = None
            self.future_temporal_encoder = None

        self.consistency_traj_encoder = TrajectoryEncoder(
            consistency_traj_steps * traj_d, hidden, act_dim,
        )
        self.validity_traj_encoder = TrajectoryEncoder(
            traj_steps * traj_d, hidden, act_dim,
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, act_dim),
            nn.ReLU(inplace=True),
        )

        if self.use_action_visual_interaction:
            self.action_to_visual_delta = nn.Sequential(
                nn.Linear(act_dim * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, img_dim),
                nn.ReLU(inplace=True),
            )
            consistency_dim = img_dim * 7 + act_dim * 2
        else:
            self.action_to_visual_delta = None
            consistency_dim = img_dim * 2 + act_dim * 2
        self.shared_fusion = nn.Sequential(
            nn.Linear(consistency_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        validity_dim = act_dim * 2
        self.validity_fusion = nn.Sequential(
            nn.Linear(validity_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Consistency 是主监督。细粒度 heads 仅保留为审计输出，默认 loss 权重为 0。
        self.consistency_head = nn.Linear(fusion_dim, 1)  # overall consistency
        self.speed_consistency_head = nn.Linear(fusion_dim, 1)  # speed consistency
        self.steering_consistency_head = nn.Linear(fusion_dim, 1)  # steering consistency
        self.progress_consistency_head = nn.Linear(fusion_dim, 1)  # progress consistency
        self.temporal_coherence_head = nn.Linear(fusion_dim, 1)  # temporal coherence
        self.validity_head = nn.Linear(fusion_dim, 1)  # driving validity

    def _encode_images(
        self,
        images: torch.Tensor,
        proj: nn.Linear,
        temporal_encoder: nn.GRU | None,
    ) -> torch.Tensor:
        """编码 (B, T, C, H, W) 图像序列为 (B, dim)"""
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        x = self.shared_backbone(x).flatten(1)
        x = proj(x)
        x = x.reshape(b, t, -1)
        if temporal_encoder is None:
            return x.mean(dim=1)
        _, hidden = temporal_encoder(x)
        return hidden[-1]

    def forward(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_hist = self._encode_images(
            history_images, self.history_proj, self.history_temporal_encoder,
        )
        z_fut = self._encode_images(
            future_images, self.future_proj, self.future_temporal_encoder,
        )
        consistency_traj = candidate_traj[:, : self.consistency_traj_steps, :]
        z_traj_consistency = self.consistency_traj_encoder(consistency_traj)
        z_traj_validity = self.validity_traj_encoder(candidate_traj)
        z_ego = self.ego_encoder(ego_state)

        mode = self.baseline_mode
        if mode in {"no_image", "ego_only", "traj_only"}:
            z_hist = torch.zeros_like(z_hist)
            z_fut = torch.zeros_like(z_fut)
        if mode in {"no_traj", "ego_only"}:
            z_traj_consistency = torch.zeros_like(z_traj_consistency)
            z_traj_validity = torch.zeros_like(z_traj_validity)
        if mode == "traj_only":
            z_ego = torch.zeros_like(z_ego)

        if self.use_action_visual_interaction:
            visual_delta = z_fut - z_hist
            visual_abs_delta = visual_delta.abs()
            assert self.action_to_visual_delta is not None
            action_delta = self.action_to_visual_delta(
                torch.cat([z_traj_consistency, z_ego], dim=-1),
            )
            action_visual_product = action_delta * visual_delta
            action_visual_gap = (action_delta - visual_delta).abs()
            z_all = torch.cat(
                [
                    z_hist,
                    z_fut,
                    visual_delta,
                    visual_abs_delta,
                    action_delta,
                    action_visual_product,
                    action_visual_gap,
                    z_traj_consistency,
                    z_ego,
                ],
                dim=-1,
            )
        else:
            z_all = torch.cat([z_hist, z_fut, z_traj_consistency, z_ego], dim=-1)
        z_shared = self.shared_fusion(z_all)
        z_validity = self.validity_fusion(torch.cat([z_traj_validity, z_ego], dim=-1))

        return {
            # Layer 2: Action一致性评估（多维度）
            "consistency_logit": self.consistency_head(z_shared).squeeze(-1),
            "speed_consistency_logit": self.speed_consistency_head(z_shared).squeeze(-1),
            "steering_consistency_logit": self.steering_consistency_head(z_shared).squeeze(-1),
            "progress_consistency_logit": self.progress_consistency_head(z_shared).squeeze(-1),
            "temporal_coherence_logit": self.temporal_coherence_head(z_shared).squeeze(-1),
            # Layer 3: 驾驶合理性评估
            "validity_logit": self.validity_head(z_validity).squeeze(-1),
        }


class GroupBatchSampler(Sampler[List[int]]):
    """Yield complete group_id batches so ranking loss sees in-group candidates."""

    def __init__(
        self,
        dataset: ConsistencyDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.epoch = 0

        groups: Dict[str, List[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            fallback = f"{sample.get('scene_name', 'unknown')}__{sample.get('timestamp_us', idx)}"
            group_id = str(sample.get("group_id") or fallback)
            groups.setdefault(group_id, []).append(idx)
        self.groups = list(groups.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)

        batches: List[List[int]] = []
        current: List[int] = []
        for group in groups:
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            if len(group) > self.batch_size:
                for start in range(0, len(group), self.batch_size):
                    chunk = group[start: start + self.batch_size]
                    if len(chunk) == self.batch_size or not self.drop_last:
                        batches.append(chunk)
                continue
            current.extend(group)
        if current and (len(current) == self.batch_size or not self.drop_last):
            batches.append(current)
        if self.world_size > 1:
            usable = len(batches) - (len(batches) % self.world_size)
            batches = batches[:usable]

        for batch_idx, batch in enumerate(batches):
            if batch_idx % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        total = 0
        current = 0
        for group in self.groups:
            group_len = len(group)
            if current and current + group_len > self.batch_size:
                total += 1
                current = 0
            if group_len > self.batch_size:
                full, rest = divmod(group_len, self.batch_size)
                total += full
                if rest and not self.drop_last:
                    total += 1
                continue
            current += group_len
        if current and (current == self.batch_size or not self.drop_last):
            total += 1
        if self.world_size > 1:
            total -= total % self.world_size
        return total // self.world_size


def compute_group_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_indices: torch.Tensor,
    dataset: ConsistencyDataset | None,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset is None or sample_indices.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    groups: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for row, sample_index in enumerate(sample_indices.detach().cpu().tolist()):
        sample = dataset.samples[int(sample_index)]
        fallback = f"{sample.get('scene_name', 'unknown')}__{sample.get('timestamp_us', sample_index)}"
        group_id = str(sample.get("group_id") or fallback)
        bucket = groups.setdefault(group_id, {"pos": [], "neg": []})
        if float(labels[row].detach().item()) > 0.5:
            bucket["pos"].append(logits[row])
        else:
            bucket["neg"].append(logits[row])

    losses: List[torch.Tensor] = []
    correct = 0.0
    count = 0.0
    for bucket in groups.values():
        if not bucket["pos"] or not bucket["neg"]:
            continue
        pos_logits = torch.stack(bucket["pos"])
        neg_logits = torch.stack(bucket["neg"])
        pairwise = margin - pos_logits[:, None] + neg_logits[None, :]
        losses.append(F.softplus(pairwise).mean())
        count += 1.0
        if pos_logits.max().item() > neg_logits.max().item():
            correct += 1.0

    if not losses:
        zero = logits.sum() * 0.0
        return zero, zero.detach(), zero.detach()
    loss = torch.stack(losses).mean()
    acc = torch.tensor(correct / max(count, 1.0), device=logits.device)
    group_count = torch.tensor(count, device=logits.device)
    return loss, acc, group_count


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
    ranking_cfg = cfg.get("ranking", {})
    lambda_ranking = float(cfg.get("lambda_group_ranking", ranking_cfg.get("loss_weight", 0.0)))
    ranking_margin = float(cfg.get("group_ranking_margin", ranking_cfg.get("margin", 0.2)))
    
    # 正样本权重
    c_pw = torch.tensor(
        cfg.get("consistency_positive_weight", cfg["positive_weight"]),
        device=device,
    )
    v_pw = torch.tensor(
        cfg.get("validity_positive_weight", cfg["positive_weight"]),
        device=device,
    )
    
    # 主任务使用逐样本加权，给难负样本更强梯度。
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
    total_ranking_loss = 0.0
    
    total_c_correct = 0.0
    total_v_correct = 0.0
    total_speed_correct = 0.0
    total_steering_correct = 0.0
    total_progress_correct = 0.0
    total_temporal_correct = 0.0
    total_ranking_correct = 0.0
    total_ranking_groups = 0.0
    
    total_samples = 0
    log_interval = int(cfg["log_interval"])
    ranking_dataset = getattr(getattr(loader, "batch_sampler", None), "dataset", None)

    if training and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if training and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)

    for step, batch in enumerate(loader, start=1):
        h_imgs = batch["history_images"].to(device, non_blocking=True)
        f_imgs = batch["future_images"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        c_labels = batch["consistency_label"].to(device, non_blocking=True)
        v_labels = batch["validity_label"].to(device, non_blocking=True)
        c_weights = batch.get("consistency_weight", torch.ones_like(c_labels)).to(
            device, non_blocking=True,
        )
        v_weights = batch.get("validity_weight", torch.ones_like(v_labels)).to(
            device, non_blocking=True,
        )
        sample_indices = batch.get("sample_index", torch.empty(0, dtype=torch.long)).to(
            device, non_blocking=True,
        )
        
        # 多维度标签（如果存在）
        speed_labels = batch.get("speed_consistency_label", c_labels).to(device, non_blocking=True)
        steering_labels = batch.get("steering_consistency_label", c_labels).to(device, non_blocking=True)
        progress_labels = batch.get("progress_consistency_label", c_labels).to(device, non_blocking=True)
        temporal_labels = batch.get("temporal_coherence_label", c_labels).to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            out = model(h_imgs, f_imgs, ego, traj)
            
            # 多维度损失计算
            raw_loss_c = F.binary_cross_entropy_with_logits(
                out["consistency_logit"], c_labels,
                pos_weight=c_pw,
                reduction="none",
            )
            raw_loss_v = F.binary_cross_entropy_with_logits(
                out["validity_logit"], v_labels,
                pos_weight=v_pw,
                reduction="none",
            )
            loss_c = (raw_loss_c * c_weights).sum() / c_weights.sum().clamp_min(1.0)
            loss_v = (raw_loss_v * v_weights).sum() / v_weights.sum().clamp_min(1.0)
            loss_speed = criterion_speed(out["speed_consistency_logit"], speed_labels)
            loss_steering = criterion_steering(out["steering_consistency_logit"], steering_labels)
            loss_progress = criterion_progress(out["progress_consistency_logit"], progress_labels)
            loss_temporal = criterion_temporal(out["temporal_coherence_logit"], temporal_labels)
            loss_ranking, ranking_acc, ranking_groups = compute_group_ranking_loss(
                logits=out["consistency_logit"],
                labels=c_labels,
                sample_indices=sample_indices,
                dataset=ranking_dataset,
                margin=ranking_margin,
            )
            
            # 加权组合
            loss = (lambda_c * loss_c + 
                   lambda_v * loss_v + 
                   lambda_speed * loss_speed +
                   lambda_steering * loss_steering +
                   lambda_progress * loss_progress +
                   lambda_temporal * loss_temporal +
                   lambda_ranking * loss_ranking)
            
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
        total_ranking_loss += loss_ranking.detach().item() * bs
        
        total_c_correct += (c_preds == c_labels).float().sum().item()
        total_v_correct += (v_preds == v_labels).float().sum().item()
        total_speed_correct += (speed_preds == speed_labels).float().sum().item()
        total_steering_correct += (steering_preds == steering_labels).float().sum().item()
        total_progress_correct += (progress_preds == progress_labels).float().sum().item()
        total_temporal_correct += (temporal_preds == temporal_labels).float().sum().item()
        total_ranking_correct += ranking_acc.detach().item() * ranking_groups.detach().item()
        total_ranking_groups += ranking_groups.detach().item()
        
        total_samples += bs

        if is_main_process() and step % log_interval == 0:
            phase = "Train" if training else "Val"
            print(
                f"[{phase}] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.detach().item():.4f} "
                f"c_loss={loss_c.detach().item():.4f} "
                f"v_loss={loss_v.detach().item():.4f} "
                f"rank_loss={loss_ranking.detach().item():.4f}",
                flush=True,
            )
        if max_steps and step >= max_steps:
            break
        if sigterm_received():
            if is_main_process():
                phase = "训练" if training else "验证"
                print(f"[WARNING] SIGTERM 中断{phase}，已完成 step={step}/{len(loader)}")
            break

    metrics = torch.tensor(
        [
            total_loss, total_c_loss, total_v_loss,
            total_speed_loss, total_steering_loss, total_progress_loss, total_temporal_loss,
            total_ranking_loss,
            total_c_correct, total_v_correct,
            total_speed_correct, total_steering_correct, total_progress_correct, total_temporal_correct,
            total_ranking_correct, total_ranking_groups,
            float(total_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    metrics = reduce_mean(metrics)
    n = max(float(metrics[16].item()), 1.0)
    rank_groups = max(float(metrics[15].item()), 1.0)
    return {
        "loss": float(metrics[0].item() / n),
        "c_loss": float(metrics[1].item() / n),
        "v_loss": float(metrics[2].item() / n),
        "speed_loss": float(metrics[3].item() / n),
        "steering_loss": float(metrics[4].item() / n),
        "progress_loss": float(metrics[5].item() / n),
        "temporal_loss": float(metrics[6].item() / n),
        "ranking_loss": float(metrics[7].item() / n),
        "c_acc": float(metrics[8].item() / n),
        "v_acc": float(metrics[9].item() / n),
        "speed_acc": float(metrics[10].item() / n),
        "steering_acc": float(metrics[11].item() / n),
        "progress_acc": float(metrics[12].item() / n),
        "temporal_acc": float(metrics[13].item() / n),
        "ranking_acc": float(metrics[14].item() / rank_groups),
        "ranking_groups": float(metrics[15].item()),
    }


def build_dataloader(cfg: Dict[str, Any], index_path: str, training: bool) -> DataLoader:
    dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=training)
    ranking_cfg = cfg.get("ranking", {})
    lambda_ranking = float(
        cfg.get("lambda_group_ranking", ranking_cfg.get("loss_weight", 0.0)),
    )
    use_group_batches = bool(ranking_cfg.get("group_batches", lambda_ranking > 0.0))
    if use_group_batches:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        batch_sampler = GroupBatchSampler(
            dataset=dataset,
            batch_size=int(cfg["batch_size"]),
            shuffle=training,
            drop_last=training,
            seed=int(cfg.get("seed", 42)),
            rank=rank,
            world_size=world_size,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(cfg["num_workers"]),
            pin_memory=True,
        )
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


def _preflight_indices(num_items: int, max_samples: int, seed: int) -> List[int]:
    if max_samples < 0 or num_items <= max_samples:
        return list(range(num_items))
    if max_samples == 1:
        return [0]
    rng = random.Random(seed)
    picked = {0, num_items - 1}
    picked.update(rng.sample(range(num_items), max_samples - len(picked)))
    return sorted(picked)


def validate_index_image_paths(
    cfg: Dict[str, Any],
    index_paths: Sequence[str],
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return

    for index_path in index_paths:
        dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=False)
        indices = _preflight_indices(len(dataset), max_samples, int(cfg.get("seed", 42)))
        bad_images: List[str] = []
        checked = 0
        for idx in indices:
            sample = dataset.samples[idx]
            image_paths = (
                dataset.selected_image_paths(
                    sample, "history_images", dataset.history_num_frames,
                )
                + dataset.selected_image_paths(
                    sample, "future_images", dataset.future_num_frames,
                )
            )
            for path in image_paths:
                checked += 1
                if not path.exists() or path.stat().st_size == 0:
                    bad_images.append(str(path))
                    if len(bad_images) >= 10:
                        break
                    continue
                try:
                    with Image.open(path) as img:
                        img.verify()
                except Exception as exc:
                    bad_images.append(f"{path} ({type(exc).__name__})")
                    if len(bad_images) >= 10:
                        break
            if len(bad_images) >= 10:
                break
        if bad_images:
            preview = "\n  ".join(bad_images)
            raise FileNotFoundError(
                f"索引图片预检失败: {index_path}\n"
                f"image_root={cfg['image_root']}\n"
                f"检查样本数={len(indices)}, 图片数={checked}\n"
                f"坏图/缺失示例:\n  {preview}"
            )
        print(
            f"[Preflight] {index_path}: "
            f"checked_samples={len(indices)} checked_images={checked}",
            flush=True,
        )


def save_checkpoint(
    work_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    best_val_loss: float,
    is_best: bool,
    tag: str = "latest",
    interrupted: bool = False,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "best_val_loss": best_val_loss,
        "interrupted": interrupted,
        "checkpoint_tag": tag,
    }
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / f"{tag}.pth")
    if tag == "latest" and is_best and not interrupted:
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
    if args.baseline_mode is not None:
        cfg["baseline_mode"] = args.baseline_mode
    if cfg.get("model_type") != "consistency":
        raise ValueError("新版训练入口只支持 model_type='consistency' 的 IAC 配置。")

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

    if is_main_process() and int(args.preflight_samples) != 0:
        validate_index_image_paths(
            cfg,
            [cfg["train_index"], cfg["val_index"]],
            int(args.preflight_samples),
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    train_loader = build_dataloader(cfg, cfg["train_index"], training=True)
    val_loader = build_dataloader(cfg, cfg["val_index"], training=False)

    model = ConsistencyCriticModel(cfg).to(device)
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
        print("NuPlan IAC Consistency Critic Training")
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
            train_metrics = run_consistency_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                cfg=cfg,
                training=True,
                max_steps=args.max_train_steps or 0,
            )
            val_metrics = run_consistency_epoch(
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
                print(
                    f"[Epoch {epoch}/{total_epochs}] "
                    f"loss={train_metrics['loss']:.4f} "
                    f"c_acc={train_metrics['c_acc']:.4f} "
                    f"v_acc={train_metrics['v_acc']:.4f} "
                    f"speed_acc={train_metrics['speed_acc']:.4f} "
                    f"steering_acc={train_metrics['steering_acc']:.4f} "
                    f"progress_acc={train_metrics['progress_acc']:.4f} "
                    f"temporal_acc={train_metrics['temporal_acc']:.4f} "
                    f"rank_acc={train_metrics['ranking_acc']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_c_acc={val_metrics['c_acc']:.4f} "
                    f"val_v_acc={val_metrics['v_acc']:.4f} "
                    f"val_rank_acc={val_metrics['ranking_acc']:.4f}"
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
                    print(f"[WARNING] 收到终止信号，保存 epoch={epoch} 的 interrupted checkpoint...")
                    save_checkpoint(
                        work_dir=work_dir,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        best_val_loss=best_val_loss,
                        is_best=False,
                        tag=f"interrupted_epoch_{epoch}",
                        interrupted=True,
                    )
                    print("[WARNING] interrupted checkpoint 已保存，训练提前退出")
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
        # 异常退出前只保存带 error 标记的 checkpoint，避免误用为正常结果。
        if is_main_process():
            try:
                print("[ERROR] 尝试保存 error checkpoint...", flush=True)
                save_checkpoint(
                    work_dir=work_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    best_val_loss=best_val_loss,
                    is_best=False,
                    tag=f"error_epoch_{epoch}",
                    interrupted=True,
                )
                print(f"[ERROR] error checkpoint 已保存至 {work_dir}/checkpoints/", flush=True)
            except Exception:
                print("[ERROR] error checkpoint 保存失败", flush=True)
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
