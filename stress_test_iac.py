#!/usr/bin/env python3
"""IAC P0 stress tests.

Cheap probes for shortcut auditing:
- reverse future frames
- mirror trajectory
- corrupt future images with black/white/noise
- shuffle trajectory order
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from train import ConsistencyCriticModel, ConsistencyDataset, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IAC shortcut stress tests")
    parser.add_argument("--config", default="configs/train_consistency_mini.py")
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--baseline-mode", default=None, choices=["full", "no_image", "ego_only", "no_traj", "traj_only"])
    parser.add_argument("--synthetic-smoke", action="store_true", help="Use random tensors instead of dataset files")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_model(cfg: Dict[str, Any], checkpoint_path: str | None, device: torch.device) -> ConsistencyCriticModel:
    model = ConsistencyCriticModel(cfg).to(device)
    if checkpoint_path is None:
        print("[WARNING] 未提供 checkpoint，使用随机初始化模型，仅用于脚本烟测")
        model.eval()
        return model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    print(f"加载 IAC checkpoint: {checkpoint_path}")
    model.eval()
    return model


def score(model: ConsistencyCriticModel, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        out = model(
            batch["history_images"].to(device),
            batch["future_images"].to(device),
            batch["ego_state"].to(device),
            batch["candidate_traj"].to(device),
        )
    return torch.sigmoid(out["consistency_logit"]).detach().cpu()


def make_batch(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "history_images": sample["history_images"].unsqueeze(0),
        "future_images": sample["future_images"].unsqueeze(0),
        "ego_state": sample["ego_state"].unsqueeze(0),
        "candidate_traj": sample["candidate_traj"].unsqueeze(0),
    }


def summarize_delta(base_scores: torch.Tensor, stress_scores: torch.Tensor) -> Dict[str, float]:
    delta = stress_scores - base_scores
    return {
        "base_mean": float(base_scores.mean().item()),
        "stress_mean": float(stress_scores.mean().item()),
        "delta_mean": float(delta.mean().item()),
        "delta_median": float(delta.median().item()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", cfg)
    if args.baseline_mode is not None:
        cfg["baseline_mode"] = args.baseline_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)

    dataset = None
    if args.synthetic_smoke:
        n = args.max_samples
        print("[INFO] 使用 synthetic smoke 数据，不读取外部图像文件")
    else:
        index_path = cfg["val_index"] if args.split == "val" else cfg["train_index"]
        dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=False)
        n = min(len(dataset), args.max_samples)

    base_scores = []
    stress_scores = {
        "future_reverse": [],
        "traj_mirror": [],
        "future_black": [],
        "future_white": [],
        "future_noise": [],
        "traj_shuffle": [],
    }

    for idx in range(n):
        if dataset is None:
            batch = {
                "history_images": torch.randn(1, int(cfg["history_num_frames"]), 3, int(cfg["image_size"]), int(cfg["image_size"])),
                "future_images": torch.randn(1, int(cfg["future_num_frames"]), 3, int(cfg["image_size"]), int(cfg["image_size"])),
                "ego_state": torch.randn(1, int(cfg["ego_state_dim"])),
                "candidate_traj": torch.randn(1, int(cfg["candidate_traj_steps"]), int(cfg["traj_dim"])),
            }
        else:
            batch = make_batch(dataset[idx])
        base = score(model, batch, device)
        base_scores.append(base)

        rev = {k: v.clone() for k, v in batch.items()}
        rev["future_images"] = torch.flip(rev["future_images"], dims=[1])
        stress_scores["future_reverse"].append(score(model, rev, device))

        mirror = {k: v.clone() for k, v in batch.items()}
        mirror["candidate_traj"][:, :, 1] *= -1.0
        if mirror["candidate_traj"].shape[-1] > 2:
            mirror["candidate_traj"][:, :, 2] *= -1.0
        stress_scores["traj_mirror"].append(score(model, mirror, device))

        black = {k: v.clone() for k, v in batch.items()}
        black["future_images"] = torch.zeros_like(black["future_images"])
        stress_scores["future_black"].append(score(model, black, device))

        white = {k: v.clone() for k, v in batch.items()}
        white["future_images"] = torch.ones_like(white["future_images"])
        stress_scores["future_white"].append(score(model, white, device))

        noise = {k: v.clone() for k, v in batch.items()}
        noise["future_images"] = torch.randn_like(noise["future_images"])
        stress_scores["future_noise"].append(score(model, noise, device))

        shuffle = {k: v.clone() for k, v in batch.items()}
        order = torch.randperm(shuffle["candidate_traj"].shape[1])
        shuffle["candidate_traj"] = shuffle["candidate_traj"][:, order, :]
        stress_scores["traj_shuffle"].append(score(model, shuffle, device))

    base_cat = torch.cat(base_scores)
    results = {
        name: summarize_delta(base_cat, torch.cat(values))
        for name, values in stress_scores.items()
    }
    results["num_samples"] = n
    results["baseline_mode"] = cfg.get("baseline_mode", "full")

    print("\nIAC Stress Test")
    print("=" * 60)
    print(f"samples={n} baseline_mode={results['baseline_mode']}")
    for name, data in results.items():
        if not isinstance(data, dict):
            continue
        print(
            f"{name}: base={data['base_mean']:.4f} "
            f"stress={data['stress_mean']:.4f} delta={data['delta_mean']:.4f}"
        )

    output_path = args.output
    if output_path is None and args.checkpoint is not None:
        output_path = str(Path(args.checkpoint).parent.parent / f"stress_{args.split}_results.json")
    if output_path is not None:
        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
