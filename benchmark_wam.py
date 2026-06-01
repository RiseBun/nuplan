#!/usr/bin/env python3
"""Benchmark WAM outputs with an IAC critic.

Input is a JSONL or PT manifest. Each sample should contain:
  - history_images: paths or nested tensor-like arrays, shape T,H,W,C or T,C,H,W
  - future_images: paths or arrays for WAM-generated future frames
  - ego_state: list[float]
  - candidate_traj: list[list[float]]

Optional fields used for reporting:
  - wam_name / model_name
  - action_type / source_type / sample_type / perturb_type
  - consistency_label / label
  - validity_label
  - group_id / anchor_id for ranking among candidates
  - perturb_magnitude / perturb_level for graded curves
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_critic import _compute_head_metrics
from train import ConsistencyCriticModel, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IAC benchmark on WAM outputs")
    parser.add_argument("--input", required=True, help="WAM output manifest: .jsonl/.json/.pt")
    parser.add_argument("--checkpoint", required=True, help="Trained IAC checkpoint")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--image-root", default=None, help="Resolve relative image paths")
    parser.add_argument("--output-dir", default="work_dirs/wam_benchmark", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--group-key", default="group_id", help="Group key for candidate ranking")
    parser.add_argument("--wam-key", default="wam_name", help="Field identifying the WAM/model")
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=None,
        help="Optional calibrated threshold for consistency pass/fail. Continuous score is always emitted.",
    )
    parser.add_argument(
        "--validity-threshold",
        type=float,
        default=None,
        help="Optional calibrated threshold for validity pass/fail.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported manifest format: {path}")

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("samples", "data", "rows"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Cannot find sample list in {path}")


def _as_tensor_image_sequence(value: Any, image_root: Path, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
    elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        frames = []
        for item in value:
            path = Path(item)
            if not path.is_absolute():
                path = image_root / path
            with Image.open(path) as img:
                image = img.convert("RGB").resize((size, size))
            arr = np.asarray(image, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        tensor = torch.stack(frames, dim=0)
    else:
        tensor = torch.tensor(value, dtype=torch.float32)

    if tensor.ndim != 4:
        raise ValueError(f"Image sequence must be 4D, got shape={tuple(tensor.shape)}")
    # Accept T,H,W,C or T,C,H,W.
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.max().item() > 2.0:
        tensor = tensor / 255.0
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return (tensor - mean[:, None, None]) / std[:, None, None]


class WAMManifestDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], cfg: Dict[str, Any], image_root: str | None) -> None:
        self.rows = rows
        self.cfg = cfg
        self.image_root = Path(image_root or cfg["image_root"])
        self.image_size = int(cfg["image_size"])
        self.history_num_frames = int(cfg["history_num_frames"])
        self.future_num_frames = int(cfg["future_num_frames"])
        self.ego_state_dim = int(cfg["ego_state_dim"])
        self.candidate_traj_steps = int(cfg["candidate_traj_steps"])
        self.traj_dim = int(cfg["traj_dim"])
        ds_cfg = cfg.get("dataset", {})
        self.mean = torch.tensor(ds_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
        self.std = torch.tensor(ds_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=torch.float32)
        self.normalize_ego = bool(ds_cfg.get("normalize_ego_state", True))
        self.normalize_traj = bool(ds_cfg.get("normalize_candidate_traj", True))
        self.normalize_mode = ds_cfg.get("normalize_mode", "tanh")
        traj_scale = ds_cfg.get("traj_scale")
        self.traj_scale = torch.tensor(traj_scale, dtype=torch.float32) if traj_scale is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def _prepare_vector(self, values: Any, length: int) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32).flatten()
        if tensor.numel() < length:
            tensor = F.pad(tensor, (0, length - tensor.numel()))
        return tensor[:length]

    def _prepare_traj(self, values: Any) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"candidate_traj must be 2D, got shape={tuple(tensor.shape)}")
        if tensor.shape[1] < self.traj_dim:
            tensor = F.pad(tensor, (0, self.traj_dim - tensor.shape[1]))
        tensor = tensor[:, : self.traj_dim]
        if tensor.shape[0] < self.candidate_traj_steps:
            tensor = F.pad(tensor, (0, 0, 0, self.candidate_traj_steps - tensor.shape[0]))
        return tensor[: self.candidate_traj_steps]

    def _select_frames(self, tensor: torch.Tensor, count: int) -> torch.Tensor:
        selected = tensor[-count:]
        if selected.shape[0] < count:
            pad = selected[:1].repeat(count - selected.shape[0], 1, 1, 1)
            selected = torch.cat([pad, selected], dim=0)
        return selected

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        future_value = row.get("future_images", row.get("generated_future_images", row.get("generated_images")))
        if future_value is None:
            raise KeyError("Sample must contain future_images/generated_future_images/generated_images")

        hist = _as_tensor_image_sequence(
            row["history_images"], self.image_root, self.image_size, self.mean, self.std,
        )
        fut = _as_tensor_image_sequence(
            future_value, self.image_root, self.image_size, self.mean, self.std,
        )
        ego = self._prepare_vector(row["ego_state"], self.ego_state_dim)
        traj = self._prepare_traj(row["candidate_traj"])
        if self.normalize_ego:
            ego = torch.tanh(ego)
        if self.normalize_traj:
            if self.normalize_mode == "linear" and self.traj_scale is not None:
                traj = traj / self.traj_scale
            else:
                traj = torch.tanh(traj)
        return {
            "history_images": self._select_frames(hist, self.history_num_frames),
            "future_images": self._select_frames(fut, self.future_num_frames),
            "ego_state": ego,
            "candidate_traj": traj,
        }


def _load_model(checkpoint_path: Path, cfg: Dict[str, Any], device: torch.device) -> ConsistencyCriticModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ConsistencyCriticModel(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def _label(row: Dict[str, Any], key: str, fallback: str = "label") -> float | None:
    if key in row:
        return float(row[key])
    if fallback in row:
        return float(row[fallback])
    return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _ndcg(labels: List[float], scores: List[float], k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = np.array(labels)[order]
    discounts = 1.0 / np.log2(np.arange(len(gains)) + 2)
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def _ranking_summary(scored: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        group = row.get(group_key) or row.get("anchor_id") or row.get("sample_id")
        if group is not None and row.get("consistency_label") is not None:
            groups[str(group)].append(row)

    top1_hits, mrrs, ndcg3, ndcg5 = [], [], [], []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        labels = [float(row["consistency_label"]) for row in rows]
        if max(labels) <= 0:
            continue
        scores = [float(row["iac_consistency"]) for row in rows]
        order = np.argsort(scores)[::-1]
        sorted_labels = np.array(labels)[order]
        top1_hits.append(float(sorted_labels[0] > 0))
        first_pos = np.where(sorted_labels > 0)[0]
        mrrs.append(1.0 / float(first_pos[0] + 1) if len(first_pos) else 0.0)
        ndcg3.append(_ndcg(labels, scores, 3))
        ndcg5.append(_ndcg(labels, scores, 5))

    return {
        "num_groups": len(groups),
        "num_ranked_groups": len(top1_hits),
        "top1_hit_rate": _mean(top1_hits),
        "mrr": _mean(mrrs),
        "ndcg@3": _mean(ndcg3),
        "ndcg@5": _mean(ndcg5),
    }


def _summary(
    scored: List[Dict[str, Any]],
    wam_key: str,
    group_key: str,
    consistency_threshold: float | None,
    validity_threshold: float | None,
) -> Dict[str, Any]:
    c_labels = [row.get("consistency_label") for row in scored]
    v_labels = [row.get("validity_label") for row in scored]
    c_scores = torch.tensor([row["iac_consistency"] for row in scored], dtype=torch.float32)
    v_scores = torch.tensor([row["iac_validity"] for row in scored], dtype=torch.float32)

    summary: Dict[str, Any] = {
        "num_samples": len(scored),
        "overall": {
            "mean_consistency": float(c_scores.mean().item()),
            "mean_validity": float(v_scores.mean().item()),
        },
        "thresholds": {
            "consistency_threshold": consistency_threshold,
            "validity_threshold": validity_threshold,
        },
        "by_wam": {},
        "by_action_type": {},
        "ranking": _ranking_summary(scored, group_key),
        "graded_perturbation_curve": {},
    }

    if all(label is not None for label in c_labels):
        labels = torch.tensor([float(label) for label in c_labels], dtype=torch.float32)
        logits = torch.logit(c_scores.clamp(1e-6, 1 - 1e-6))
        summary["overall"]["consistency_binary"] = _compute_head_metrics(logits, labels)
    if all(label is not None for label in v_labels):
        labels = torch.tensor([float(label) for label in v_labels], dtype=torch.float32)
        logits = torch.logit(v_scores.clamp(1e-6, 1 - 1e-6))
        summary["overall"]["validity_binary"] = _compute_head_metrics(logits, labels)

    for key_name, output_key in ((wam_key, "by_wam"), ("action_type", "by_action_type")):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in scored:
            value = row.get(key_name) or row.get("model_name") or row.get("source_type") or row.get("sample_type") or "unknown"
            groups[str(value)].append(row)
        for value, rows in groups.items():
            summary[output_key][value] = {
                "count": len(rows),
                "mean_consistency": _mean(row["iac_consistency"] for row in rows),
                "mean_validity": _mean(row["iac_validity"] for row in rows),
            }
            if consistency_threshold is not None:
                summary[output_key][value]["consistency_pass_rate"] = _mean(
                    float(row["consistency_pass"]) for row in rows
                )
            if validity_threshold is not None:
                summary[output_key][value]["validity_pass_rate"] = _mean(
                    float(row["validity_pass"]) for row in rows
                )

    if consistency_threshold is not None:
        summary["overall"]["consistency_pass_rate"] = _mean(
            float(row["consistency_pass"]) for row in scored
        )
    if validity_threshold is not None:
        summary["overall"]["validity_pass_rate"] = _mean(
            float(row["validity_pass"]) for row in scored
        )

    graded: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if row.get("perturb_magnitude") is None:
            continue
        ptype = row.get("perturb_type") or row.get("action_type") or "perturb"
        level = row.get("perturb_level", "unknown")
        graded[f"{ptype}:{level}"].append(row)
    for key, rows in graded.items():
        summary["graded_perturbation_curve"][key] = {
            "count": len(rows),
            "mean_consistency": _mean(row["iac_consistency"] for row in rows),
            "mean_perturb_magnitude": _mean(float(row["perturb_magnitude"]) for row in rows),
        }
    return summary


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config) if args.config else checkpoint["config"]
    rows = _load_rows(Path(args.input))
    if args.max_samples:
        rows = rows[: args.max_samples]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(Path(args.checkpoint), cfg, device)
    dataset = WAMManifestDataset(rows, cfg, args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    scored: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["history_images"].to(device, non_blocking=True),
                batch["future_images"].to(device, non_blocking=True),
                batch["ego_state"].to(device, non_blocking=True),
                batch["candidate_traj"].to(device, non_blocking=True),
            )
            c_scores = torch.sigmoid(out["consistency_logit"]).cpu().tolist()
            v_scores = torch.sigmoid(out["validity_logit"]).cpu().tolist()
            for i, (c_score, v_score) in enumerate(zip(c_scores, v_scores)):
                row = dict(rows[offset + i])
                row["iac_consistency"] = float(c_score)
                row["iac_validity"] = float(v_score)
                if args.consistency_threshold is not None:
                    row["consistency_threshold"] = float(args.consistency_threshold)
                    row["consistency_pass"] = bool(c_score >= args.consistency_threshold)
                if args.validity_threshold is not None:
                    row["validity_threshold"] = float(args.validity_threshold)
                    row["validity_pass"] = bool(v_score >= args.validity_threshold)
                c_label = _label(row, "consistency_label")
                v_label = _label(row, "validity_label")
                if c_label is not None:
                    row["consistency_label"] = c_label
                if v_label is not None:
                    row["validity_label"] = v_label
                scored.append(row)
            offset += len(c_scores)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "wam_iac_scores.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summary(
        scored,
        args.wam_key,
        args.group_key,
        args.consistency_threshold,
        args.validity_threshold,
    )
    summary["input"] = str(args.input)
    summary["checkpoint"] = str(args.checkpoint)
    summary_path = out_dir / "wam_iac_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nWAM IAC Benchmark")
    print("=" * 60)
    print(f"samples={summary['num_samples']}")
    print(f"mean_consistency={summary['overall']['mean_consistency']:.4f}")
    print(f"mean_validity={summary['overall']['mean_validity']:.4f}")
    if "consistency_pass_rate" in summary["overall"]:
        print(f"consistency_pass_rate={summary['overall']['consistency_pass_rate']:.4f}")
    if "validity_pass_rate" in summary["overall"]:
        print(f"validity_pass_rate={summary['overall']['validity_pass_rate']:.4f}")
    print(f"scores={scored_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
