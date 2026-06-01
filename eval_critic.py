#!/usr/bin/env python3
"""IAC Consistency Critic 模型评估脚本

用法:
    # 评估 Consistency Critic 模型
    python eval_critic.py --checkpoint work_dirs/iac_full/checkpoints/best.pth

    # 限制评估样本数
    python eval_critic.py --checkpoint work_dirs/iac_full/checkpoints/best.pth --max-samples 100
    
    # Ranking 评估（需要索引中包含 ranking_groups）
    python eval_critic.py --checkpoint work_dirs/iac_full/checkpoints/best.pth --eval-ranking
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from train import ConsistencyDataset, ConsistencyCriticModel, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Critic 模型")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint 文件路径 (.pth)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认从 checkpoint 中读取）",
    )
    parser.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="评估数据集划分 (默认: val)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="评估 batch size (默认: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="评估 DataLoader worker 数。默认 0，更稳但可能更慢。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多评估样本数，0 表示全部 (默认: 0)",
    )
    parser.add_argument(
        "--eval-ranking",
        action="store_true",
        help="是否评估 ranking 能力（NDCG, MRR, Top-k）",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["full", "no_image", "ego_only", "no_traj", "traj_only"],
        default=None,
        help="覆盖 checkpoint/config 中的 P0 baseline mode",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="输出文件前缀，默认 eval_<split>。可用于 quick eval 避免覆盖正式结果。",
    )
    return parser.parse_args()


def _compute_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if labels.numel() == 0:
        return 0.0
    ece = torch.tensor(0.0)
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.any():
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            ece += mask.float().mean() * torch.abs(conf - acc)
    return float(ece.item())


def _safe_average_precision(labels: torch.Tensor, probs: torch.Tensor) -> float | None:
    if labels.numel() == 0 or labels.unique().numel() < 2:
        return None
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels.numpy(), probs.numpy()))
    except ImportError:
        order = torch.argsort(probs, descending=True)
        sorted_labels = labels[order]
        positives = sorted_labels.sum().item()
        if positives <= 0:
            return None
        tp = torch.cumsum(sorted_labels, dim=0)
        ranks = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
        precision_at_k = tp / ranks
        return float((precision_at_k * sorted_labels).sum().item() / positives)


def _compute_head_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Any]:
    """计算单个 head 的详细指标"""
    probs = torch.sigmoid(logits)
    pos_mask = labels == 1.0
    neg_mask = labels == 0.0
    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()

    def metrics_at_threshold(threshold: float) -> Dict[str, Any]:
        preds = (probs >= threshold).float()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None
            and (precision + recall) > 0
            else None
        )
        accuracy = (preds == labels).float().mean().item()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        balanced_accuracy = (
            (recall + tnr) / 2
            if recall is not None and tnr is not None
            else None
        )
        return {
            "threshold": float(threshold),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tnr": tnr,
            "fpr": fpr,
            "balanced_accuracy": balanced_accuracy,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    fixed = metrics_at_threshold(0.5)

    best_f1 = None
    best_balanced = None
    recall_thresholds: Dict[str, Any] = {}
    if num_pos > 0 and num_neg > 0:
        # 用固定网格估计 operating point，避免全量验证集上逐唯一概率扫描过慢。
        candidate_thresholds = torch.linspace(0.0, 1.0, steps=201).tolist()
        candidate_thresholds.append(0.5)
        for threshold in sorted(set(float(t) for t in candidate_thresholds)):
            current = metrics_at_threshold(threshold)
            f1 = current["f1_score"]
            if f1 is not None and (
                best_f1 is None or f1 > best_f1["f1_score"]
            ):
                best_f1 = current
            bal = current["balanced_accuracy"]
            if bal is not None and (
                best_balanced is None or bal > best_balanced["balanced_accuracy"]
            ):
                best_balanced = current
        for target_recall in (0.80, 0.90, 0.95):
            valid = []
            for threshold in sorted(set(float(t) for t in candidate_thresholds)):
                current = metrics_at_threshold(threshold)
                recall = current["recall"]
                if recall is not None and recall >= target_recall:
                    valid.append(current)
            if valid:
                selected = max(
                    valid,
                    key=lambda item: (
                        item["tnr"] if item["tnr"] is not None else -1.0,
                        item["precision"] if item["precision"] is not None else -1.0,
                    ),
                )
                recall_thresholds[f"recall>={target_recall:.2f}"] = selected

    # AUC 计算
    auc: float | None = None
    pr_auc: float | None = None
    if num_pos > 0 and num_neg > 0:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(
                labels.numpy(), probs.numpy(),
            )
        except ImportError:
            # 简易 AUC: 正样本概率 > 负样本概率的比例
            pos_p = probs[pos_mask]
            neg_p = probs[neg_mask]
            comparisons = (
                pos_p.unsqueeze(1) > neg_p.unsqueeze(0)
            ).float().mean().item()
            auc = comparisons
        pr_auc = _safe_average_precision(labels, probs)

    pos_probs = probs[pos_mask].numpy() if num_pos > 0 else np.array([])
    neg_probs = probs[neg_mask].numpy() if num_neg > 0 else np.array([])

    return {
        "num_positive": int(num_pos),
        "num_negative": int(num_neg),
        "accuracy": fixed["accuracy"],
        "precision": fixed["precision"],
        "recall": fixed["recall"],
        "f1_score": fixed["f1_score"],
        "tnr": fixed["tnr"],
        "fpr": fixed["fpr"],
        "balanced_accuracy": fixed["balanced_accuracy"],
        "auc": auc,
        "pr_auc": pr_auc,
        "ece": _compute_ece(probs, labels),
        "tp": fixed["tp"],
        "fp": fixed["fp"],
        "fn": fixed["fn"],
        "tn": fixed["tn"],
        "pos_prob_mean": float(pos_probs.mean()) if len(pos_probs) > 0 else 0.0,
        "neg_prob_mean": float(neg_probs.mean()) if len(neg_probs) > 0 else 0.0,
        "fixed_threshold": fixed,
        "best_f1_threshold": best_f1,
        "best_balanced_accuracy_threshold": best_balanced,
        "recall_operating_points": recall_thresholds,
    }


def evaluate_consistency(
    model: nn.Module,
    dataset: "ConsistencyDataset",
    device: torch.device,
    batch_size: int,
    max_samples: int,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """评估 Consistency Critic 模型，返回双头指标和 per-source-type 分组统计"""
    from torch.utils.data import DataLoader
    from collections import defaultdict

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    model.eval()

    all_c_logits: list = []
    all_v_logits: list = []
    all_c_labels: list = []
    all_v_labels: list = []
    all_source_types: list = []
    sample_meta: list = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            remaining = max_samples - total_samples if max_samples else len(
                batch["consistency_label"]
            )
            if remaining <= 0:
                break
            batch_limit = min(len(batch["consistency_label"]), remaining)
            h_imgs = batch["history_images"].to(device, non_blocking=True)
            f_imgs = batch["future_images"].to(device, non_blocking=True)
            ego = batch["ego_state"].to(device, non_blocking=True)
            traj = batch["candidate_traj"].to(device, non_blocking=True)
            c_labels = batch["consistency_label"][:batch_limit]
            v_labels = batch["validity_label"][:batch_limit]

            out = model(h_imgs, f_imgs, ego, traj)
            all_c_logits.append(out["consistency_logit"][:batch_limit].cpu())
            all_v_logits.append(out["validity_logit"][:batch_limit].cpu())
            all_c_labels.append(c_labels)
            all_v_labels.append(v_labels)

            # 收集 source_type
            start = batch_idx * batch_size
            end = min(start + batch_limit, len(dataset))
            for i in range(start, end):
                st = dataset.samples[i].get("source_type", "unknown")
                all_source_types.append(st)
                sample_meta.append(dataset.samples[i])

            total_samples += batch_limit
            if (batch_idx + 1) % 20 == 0:
                print(
                    f"[Eval] step={batch_idx + 1}/{len(loader)} "
                    f"samples={total_samples}",
                    flush=True,
                )
            if max_samples and total_samples >= max_samples:
                break

    c_logits = torch.cat(all_c_logits)[:total_samples]
    v_logits = torch.cat(all_v_logits)[:total_samples]
    c_labels = torch.cat(all_c_labels)[:total_samples]
    v_labels = torch.cat(all_v_labels)[:total_samples]
    source_types = all_source_types[:total_samples]
    sample_meta = sample_meta[:total_samples]

    # 整体指标
    consistency_metrics = _compute_head_metrics(c_logits, c_labels)
    validity_metrics = _compute_head_metrics(v_logits, v_labels)

    # per-source-type 分组指标
    source_groups: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"c_logits": [], "v_logits": [], "c_labels": [], "v_labels": []},
    )
    for i, st in enumerate(source_types):
        source_groups[st]["c_logits"].append(c_logits[i])
        source_groups[st]["v_logits"].append(v_logits[i])
        source_groups[st]["c_labels"].append(c_labels[i])
        source_groups[st]["v_labels"].append(v_labels[i])

    per_source: Dict[str, Dict] = {}
    for st, data in sorted(source_groups.items()):
        st_c_logits = torch.stack(data["c_logits"])
        st_c_labels = torch.stack(data["c_labels"])
        st_v_logits = torch.stack(data["v_logits"])
        st_v_labels = torch.stack(data["v_labels"])
        per_source[st] = {
            "count": len(data["c_logits"]),
            "consistency": _compute_head_metrics(st_c_logits, st_c_labels),
            "validity": _compute_head_metrics(st_v_logits, st_v_labels),
        }

    negative_recall_by_type = {
        st: data["consistency"]["tnr"]
        for st, data in per_source.items()
        if data["consistency"]["num_negative"] > 0
    }

    c_probs = torch.sigmoid(c_logits)
    graded_groups: Dict[str, Dict[str, list]] = defaultdict(lambda: {"probs": [], "magnitudes": []})
    for i, meta in enumerate(sample_meta):
        if not str(meta.get("source_type", "")).startswith("perturb_"):
            continue
        ptype = meta.get("perturb_type", meta.get("source_type", "perturb").replace("perturb_", ""))
        level = meta.get("perturb_level", "unknown")
        key = f"{ptype}:{level}"
        graded_groups[key]["probs"].append(float(c_probs[i].item()))
        if "perturb_magnitude" in meta:
            graded_groups[key]["magnitudes"].append(float(meta["perturb_magnitude"]))

    graded_curve = {}
    for key, data in sorted(graded_groups.items()):
        probs = data["probs"]
        mags = data["magnitudes"]
        graded_curve[key] = {
            "count": len(probs),
            "mean_consistency_prob": float(np.mean(probs)) if probs else 0.0,
            "mean_perturb_magnitude": float(np.mean(mags)) if mags else None,
        }

    return {
        "total_samples": total_samples,
        "consistency": consistency_metrics,
        "validity": validity_metrics,
        "per_source_type": per_source,
        "negative_recall_by_type": negative_recall_by_type,
        "graded_perturbation_curve": graded_curve,
    }


def _print_head_metrics(name: str, m: Dict[str, Any], indent: str = "  ") -> None:
    """打印单个 head 的评估指标"""
    print(f"{indent}[{name}]")
    print(f"{indent}  正/负样本数: {m['num_positive']} / {m['num_negative']}")
    print(f"{indent}  固定阈值: 0.5000")
    print(f"{indent}  Accuracy:  {m['accuracy']:.4f}")
    if m['num_positive'] > 0:
        p = m['precision']
        r = m['recall']
        f1 = m['f1_score']
        print(f"{indent}  Precision: {p:.4f}" if p is not None else f"{indent}  Precision: N/A")
        print(f"{indent}  Recall:    {r:.4f}" if r is not None else f"{indent}  Recall:    N/A")
        print(f"{indent}  F1 Score:  {f1:.4f}" if f1 is not None else f"{indent}  F1 Score:  N/A")
    else:
        print(f"{indent}  (无正样本，Precision/Recall/F1 不适用)")
    if m['num_negative'] > 0:
        tnr = m.get('tnr')
        fpr = m.get('fpr')
        print(f"{indent}  TNR:       {tnr:.4f}" if tnr is not None else f"{indent}  TNR:       N/A")
        print(f"{indent}  FPR:       {fpr:.4f}" if fpr is not None else f"{indent}  FPR:       N/A")
    else:
        print(f"{indent}  (无负样本，TNR/FPR 不适用)")
    if m.get('auc') is not None:
        print(f"{indent}  AUC:       {m['auc']:.4f}")
    else:
        print(f"{indent}  AUC:       N/A (需要同时有正负样本)")
    if m.get('pr_auc') is not None:
        print(f"{indent}  PR-AUC:    {m['pr_auc']:.4f}")
    else:
        print(f"{indent}  PR-AUC:    N/A (需要同时有正负样本)")
    print(f"{indent}  ECE:       {m.get('ece', 0.0):.4f}")
    print(f"{indent}  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}")
    print(f"{indent}  正样本概率均值: {m['pos_prob_mean']:.4f}")
    print(f"{indent}  负样本概率均值: {m['neg_prob_mean']:.4f}")
    if m.get("best_f1_threshold"):
        best = m["best_f1_threshold"]
        print(
            f"{indent}  Best-F1阈值: {best['threshold']:.4f} "
            f"F1={best['f1_score']:.4f} "
            f"Recall={best['recall']:.4f} "
            f"TNR={best['tnr']:.4f}"
        )
    if m.get("best_balanced_accuracy_threshold"):
        best = m["best_balanced_accuracy_threshold"]
        print(
            f"{indent}  Best-BalAcc阈值: {best['threshold']:.4f} "
            f"BalAcc={best['balanced_accuracy']:.4f} "
            f"Recall={best['recall']:.4f} "
            f"TNR={best['tnr']:.4f}"
        )
    if m.get("recall_operating_points"):
        print(f"{indent}  Recall operating points:")
        for key, point in m["recall_operating_points"].items():
            precision = point["precision"]
            precision_text = f"{precision:.4f}" if precision is not None else "N/A"
            print(
                f"{indent}    {key}: threshold={point['threshold']:.4f} "
                f"precision={precision_text} "
                f"tnr={point['tnr']:.4f}"
            )


def _format_source_line(m: Dict[str, Any]) -> str:
    """根据子集的正负样本情况，智能选择展示的指标"""
    parts = [f"acc={m['accuracy']:.4f}"]
    if m['num_positive'] > 0 and m['f1_score'] is not None:
        parts.append(f"f1={m['f1_score']:.4f}")
    if m['num_negative'] > 0 and m.get('tnr') is not None:
        parts.append(f"tnr={m['tnr']:.4f}")
    if m.get('auc') is not None:
        parts.append(f"auc={m['auc']:.4f}")
    return " ".join(parts)


def compute_ranking_metrics(
    model: nn.Module,
    dataset: "ConsistencyDataset",
    device: torch.device,
    batch_size: int = 32,
    max_samples: int = 0,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """评估 Consistency Critic 的 Ranking 能力
    
    对于同一 history 的多个候选轨迹，评估模型是否能正确排序
    Metrics: NDCG@k, MRR, Top-1 Hit Rate
    """
    from torch.utils.data import DataLoader, Subset
    
    # 按同一个 anchor/group 分组，而不是把整段 scene 混在一起。
    # build_consistency_index.py 会写入 group_id；旧索引用 scene+timestamp 兜底。
    scene_groups = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        scene_name = sample.get("scene_name", "unknown")
        timestamp = sample.get("timestamp_us", idx)
        group_id = sample.get("group_id") or f"{scene_name}__{timestamp}"
        scene_groups[str(group_id)].append({
            "index": idx,
            "timestamp": timestamp,
            "consistency_label": sample.get("consistency_label", 0),
            "validity_label": sample.get("validity_label", 0),
        })
    
    # 过滤出有多个候选且至少包含一个正样本的 group。
    candidate_groups = [
        (scene, samples)
        for scene, samples in scene_groups.items()
        if len(samples) >= 2
        and any(float(sample.get("consistency_label", 0)) > 0 for sample in samples)
    ]
    candidate_groups.sort(
        key=lambda item: min(sample["index"] for sample in item[1]),
    )

    selected_groups = []
    selected_indices = []
    selected_count = 0
    for scene, samples in candidate_groups:
        group_indices = [sample["index"] for sample in samples]
        if max_samples > 0 and selected_count + len(group_indices) > max_samples:
            if not selected_groups:
                selected_groups.append((scene, samples))
                selected_indices.extend(group_indices)
            break
        selected_groups.append((scene, samples))
        selected_indices.extend(group_indices)
        selected_count += len(group_indices)
    
    if not selected_groups:
        print("[WARNING] 没有找到多候选场景，跳过 ranking 评估")
        return {}
    
    print(
        f"\n[Ranking Evaluation] groups={len(selected_groups)}/"
        f"{len(candidate_groups)} samples={len(selected_indices)}"
    )
    
    model.eval()
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    score_by_index: Dict[int, float] = {}
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            out = model(
                batch["history_images"].to(device, non_blocking=True),
                batch["future_images"].to(device, non_blocking=True),
                batch["ego_state"].to(device, non_blocking=True),
                batch["candidate_traj"].to(device, non_blocking=True),
            )
            scores = torch.sigmoid(out["consistency_logit"]).detach().cpu().tolist()
            for i, score in enumerate(scores):
                score_by_index[selected_indices[offset + i]] = float(score)
            offset += len(scores)
            if batch_idx % 20 == 0 or batch_idx == len(loader):
                print(
                    f"[Ranking] scored_batches={batch_idx}/{len(loader)} "
                    f"samples={offset}",
                    flush=True,
                )

    all_ndcg_3 = []
    all_ndcg_5 = []
    all_mrr = []
    all_top1_hit = []
    
    def compute_ndcg(scores_list, relevance_list, k):
        if len(scores_list) < 2:
            return 0.0
        sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
        sorted_relevances = [rel for _, rel in sorted_pairs[:k]]
        dcg = sum(
            rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevances)
        )
        ideal_relevances = sorted(relevance_list, reverse=True)[:k]
        idcg = sum(
            rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances)
        )
        return dcg / idcg if idcg > 0 else 0.0

    def compute_mrr(scores_list, relevance_list):
        if len(scores_list) < 2:
            return 0.0
        sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
        for i, (_, rel) in enumerate(sorted_pairs):
            if rel == 1:
                return 1.0 / (i + 1)
        return 0.0

    def compute_top1_hit(scores_list, relevance_list):
        if len(scores_list) < 2:
            return 0.0
        best_idx = np.argmax(scores_list)
        return 1.0 if relevance_list[best_idx] == 1 else 0.0

    for scene_idx, (scene_name, candidates) in enumerate(selected_groups, start=1):
        scores = [score_by_index[cand["index"]] for cand in candidates]
        relevances = [float(cand["consistency_label"]) for cand in candidates]
        all_ndcg_3.append(compute_ndcg(scores, relevances, k=3))
        all_ndcg_5.append(compute_ndcg(scores, relevances, k=5))
        all_mrr.append(compute_mrr(scores, relevances))
        all_top1_hit.append(compute_top1_hit(scores, relevances))
        if scene_idx % 200 == 0 or scene_idx == len(selected_groups):
            print(
                f"[Ranking] evaluated_groups={scene_idx}/{len(selected_groups)}",
                flush=True,
            )
    
    return {
        "ndcg@3": float(np.mean(all_ndcg_3)) if all_ndcg_3 else 0.0,
        "ndcg@5": float(np.mean(all_ndcg_5)) if all_ndcg_5 else 0.0,
        "mrr": float(np.mean(all_mrr)) if all_mrr else 0.0,
        "top1_hit_rate": float(np.mean(all_top1_hit)) if all_top1_hit else 0.0,
        "num_scenes": len(selected_groups),
        "num_ranked_groups": len(all_top1_hit),
        "num_scored_samples": len(selected_indices),
        "total_candidate_groups": len(candidate_groups),
        "max_samples": int(max_samples),
    }


def main() -> None:
    args = parse_args()

    # 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    print(f"加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 加载配置
    if args.config:
        cfg = load_config(args.config)
    elif "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        raise ValueError("Checkpoint 中无 config，请用 --config 指定配置文件")
    if args.baseline_mode is not None:
        cfg["baseline_mode"] = args.baseline_mode

    epoch = checkpoint.get("epoch", "?")
    best_val_loss = checkpoint.get("best_val_loss", "?")
    model_type = cfg.get("model_type")
    if model_type != "consistency":
        raise ValueError(
            "新版 eval_critic.py 只支持 IAC checkpoint "
            "(config.model_type 必须为 'consistency')。旧 critic checkpoint 请重新训练新版 IAC。"
        )
    print(f"Checkpoint 信息: epoch={epoch}, best_val_loss={best_val_loss}")
    print("模型类型: consistency")
    print(f"Baseline mode: {cfg.get('baseline_mode', 'full')}")

    # 构建模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConsistencyCriticModel(cfg).to(device)
    
    model_state = checkpoint["model"]
    model.load_state_dict(model_state, strict=True)
    print("  权重严格匹配")
    
    print(f"模型加载完成，设备: {device}")

    # 构建数据集
    index_key = "val_index" if args.split == "val" else "train_index"
    index_path = cfg[index_key]
    print(f"数据集: {args.split} ({index_path})")

    dataset = ConsistencyDataset(
        index_path=index_path, cfg=cfg, training=False,
    )
    print(f"样本总数: {len(dataset)}")

    # 评估
    print("\n开始评估...")

    metrics = evaluate_consistency(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    print("\n" + "=" * 60)
    print("IAC Consistency Critic 评估结果")
    print("=" * 60)
    print(f"  总样本数: {metrics['total_samples']}")
    _print_head_metrics("Consistency Head", metrics["consistency"])
    _print_head_metrics("Validity Head", metrics["validity"])

    if metrics.get("per_source_type"):
        print("\n  [Per Source Type]")
        for st, st_data in metrics["per_source_type"].items():
            print(f"    --- {st} (n={st_data['count']}) ---")
            c = st_data["consistency"]
            v = st_data["validity"]
            print(f"      consistency: {_format_source_line(c)}")
            print(f"      validity:    {_format_source_line(v)}")
    if metrics.get("negative_recall_by_type"):
        print("\n  [Negative Recall / TNR by Type]")
        for st, value in metrics["negative_recall_by_type"].items():
            print(f"    {st}: {value:.4f}" if value is not None else f"    {st}: N/A")
    if metrics.get("graded_perturbation_curve"):
        print("\n  [Graded Perturbation Curve]")
        for key, data in metrics["graded_perturbation_curve"].items():
            print(
                f"    {key}: n={data['count']} "
                f"mean_prob={data['mean_consistency_prob']:.4f} "
                f"mean_mag={data['mean_perturb_magnitude']}"
            )
    print("=" * 60)

    # 保存结果到 JSON
    output_prefix = args.output_prefix or f"eval_{args.split}"
    result_path = ckpt_path.parent.parent / f"{output_prefix}_results.json"
    
    # 如果启用 ranking 评估
    if args.eval_ranking:
        print("\n" + "=" * 60)
        print("开始 Ranking 评估...")
        print("=" * 60)
        ranking_metrics = compute_ranking_metrics(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
        )
        
        if ranking_metrics:
            print("\n[Ranking Metrics]")
            print(f"  场景数: {ranking_metrics['num_scenes']}")
            print(f"  NDCG@3:  {ranking_metrics['ndcg@3']:.4f}")
            print(f"  NDCG@5:  {ranking_metrics['ndcg@5']:.4f}")
            print(f"  MRR:     {ranking_metrics['mrr']:.4f}")
            print(f"  Top-1 Hit Rate: {ranking_metrics['top1_hit_rate']:.4f}")
            print("=" * 60)
            
            # 合并到结果中
            metrics["ranking"] = ranking_metrics
    
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_path}")

    summary_path = ckpt_path.parent.parent / f"{output_prefix}_summary.json"
    summary = {
        "total_samples": metrics["total_samples"],
        "baseline_mode": cfg.get("baseline_mode", "full"),
        "consistency": {
            k: metrics["consistency"].get(k)
            for k in (
                "accuracy", "auc", "pr_auc", "ece", "f1_score", "tnr", "fpr",
                "balanced_accuracy", "best_f1_threshold",
                "best_balanced_accuracy_threshold", "recall_operating_points",
            )
        },
        "validity": {
            k: metrics["validity"].get(k)
            for k in (
                "accuracy", "auc", "pr_auc", "ece", "f1_score", "tnr", "fpr",
                "balanced_accuracy", "best_f1_threshold",
                "best_balanced_accuracy_threshold", "recall_operating_points",
            )
        },
        "negative_recall_by_type": metrics.get("negative_recall_by_type", {}),
        "graded_perturbation_curve": metrics.get("graded_perturbation_curve", {}),
        "ranking": metrics.get("ranking", {}),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
