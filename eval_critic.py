#!/usr/bin/env python3
"""Critic / Consistency Critic 模型评估脚本

用法:
    # 评估原始 Critic 模型
    python eval_critic.py --checkpoint work_dirs/critic_mini_v1/checkpoints/best.pth

    # 评估 Consistency Critic 模型 (自动检测 model_type)
    python eval_critic.py --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth

    # 限制评估样本数
    python eval_critic.py --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth --max-samples 100
    
    # Ranking 评估（需要索引中包含 ranking_groups）
    python eval_critic.py --checkpoint work_dirs/consistency_mini_v2/checkpoints/best.pth --eval-ranking
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from train import (
    CriticJsonlDataset, CriticModel,
    ConsistencyDataset, ConsistencyCriticModel,
    load_config,
)


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
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataset: CriticJsonlDataset,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> Dict[str, Any]:
    """在数据集上评估模型，返回详细指标"""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            ego_state = batch["ego_state"].to(device, non_blocking=True)
            candidate_traj = batch["candidate_traj"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images, ego_state, candidate_traj)
            loss = criterion(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            if max_samples and total_samples >= max_samples:
                break

    logits_cat = torch.cat(all_logits)[:total_samples]
    labels_cat = torch.cat(all_labels)[:total_samples]
    probs = torch.sigmoid(logits_cat)
    preds = (probs >= 0.5).float()

    # 基础指标
    accuracy = (preds == labels_cat).float().mean().item()
    avg_loss = total_loss / max(total_samples, 1)

    # 正负样本分析
    pos_mask = labels_cat == 1.0
    neg_mask = labels_cat == 0.0
    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()

    # 精确率和召回率
    tp = ((preds == 1) & (labels_cat == 1)).sum().item()
    fp = ((preds == 1) & (labels_cat == 0)).sum().item()
    fn = ((preds == 0) & (labels_cat == 1)).sum().item()
    tn = ((preds == 0) & (labels_cat == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # 正样本预测概率分布
    pos_probs = probs[pos_mask].numpy() if num_pos > 0 else np.array([])
    neg_probs = probs[neg_mask].numpy() if num_neg > 0 else np.array([])

    return {
        "total_samples": total_samples,
        "num_positive": int(num_pos),
        "num_negative": int(num_neg),
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pos_prob_mean": float(pos_probs.mean()) if len(pos_probs) > 0 else 0.0,
        "neg_prob_mean": float(neg_probs.mean()) if len(neg_probs) > 0 else 0.0,
    }


def _compute_head_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Any]:
    """计算单个 head 的详细指标"""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    pos_mask = labels == 1.0
    neg_mask = labels == 0.0
    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()

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

    # TNR / FPR
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    # AUC 计算
    auc: float | None = None
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

    pos_probs = probs[pos_mask].numpy() if num_pos > 0 else np.array([])
    neg_probs = probs[neg_mask].numpy() if num_neg > 0 else np.array([])

    return {
        "num_positive": int(num_pos),
        "num_negative": int(num_neg),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tnr": tnr,
        "fpr": fpr,
        "auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pos_prob_mean": float(pos_probs.mean()) if len(pos_probs) > 0 else 0.0,
        "neg_prob_mean": float(neg_probs.mean()) if len(neg_probs) > 0 else 0.0,
    }


def evaluate_consistency(
    model: nn.Module,
    dataset: "ConsistencyDataset",
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> Dict[str, Any]:
    """评估 Consistency Critic 模型，返回双头指标和 per-source-type 分组统计"""
    from torch.utils.data import DataLoader
    from collections import defaultdict

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    model.eval()

    all_c_logits: list = []
    all_v_logits: list = []
    all_c_labels: list = []
    all_v_labels: list = []
    all_source_types: list = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            h_imgs = batch["history_images"].to(device, non_blocking=True)
            f_imgs = batch["future_images"].to(device, non_blocking=True)
            ego = batch["ego_state"].to(device, non_blocking=True)
            traj = batch["candidate_traj"].to(device, non_blocking=True)
            c_labels = batch["consistency_label"]
            v_labels = batch["validity_label"]

            out = model(h_imgs, f_imgs, ego, traj)
            all_c_logits.append(out["consistency_logit"].cpu())
            all_v_logits.append(out["validity_logit"].cpu())
            all_c_labels.append(c_labels)
            all_v_labels.append(v_labels)

            # 收集 source_type
            start = batch_idx * batch_size
            end = min(start + len(c_labels), len(dataset))
            for i in range(start, end):
                st = dataset.samples[i].get("source_type", "unknown")
                all_source_types.append(st)

            total_samples += len(c_labels)
            if max_samples and total_samples >= max_samples:
                break

    c_logits = torch.cat(all_c_logits)[:total_samples]
    v_logits = torch.cat(all_v_logits)[:total_samples]
    c_labels = torch.cat(all_c_labels)[:total_samples]
    v_labels = torch.cat(all_v_labels)[:total_samples]
    source_types = all_source_types[:total_samples]

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

    return {
        "total_samples": total_samples,
        "consistency": consistency_metrics,
        "validity": validity_metrics,
        "per_source_type": per_source,
    }


def _print_head_metrics(name: str, m: Dict[str, Any], indent: str = "  ") -> None:
    """打印单个 head 的评估指标"""
    print(f"{indent}[{name}]")
    print(f"{indent}  正/负样本数: {m['num_positive']} / {m['num_negative']}")
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
    print(f"{indent}  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}")
    print(f"{indent}  正样本概率均值: {m['pos_prob_mean']:.4f}")
    print(f"{indent}  负样本概率均值: {m['neg_prob_mean']:.4f}")


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
) -> Dict[str, Any]:
    """评估 Consistency Critic 的 Ranking 能力
    
    对于同一 history 的多个候选轨迹，评估模型是否能正确排序
    Metrics: NDCG@k, MRR, Top-1 Hit Rate
    """
    from torch.utils.data import DataLoader
    
    # 按 scene 分组
    scene_groups = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        scene_name = sample.get("scene_name", "unknown")
        timestamp = sample.get("timestamp_us", idx)
        scene_groups[scene_name].append({
            "index": idx,
            "timestamp": timestamp,
            "consistency_label": sample.get("consistency_label", 0),
            "validity_label": sample.get("validity_label", 0),
        })
    
    # 过滤出有多个候选的 scenes
    multi_candidate_scenes = {
        scene: samples for scene, samples in scene_groups.items()
        if len(samples) >= 2
    }
    
    if not multi_candidate_scenes:
        print("[WARNING] 没有找到多候选场景，跳过 ranking 评估")
        return {}
    
    print(f"\n[Ranking Evaluation] 找到 {len(multi_candidate_scenes)} 个多候选场景")
    
    model.eval()
    all_ndcg_3 = []
    all_ndcg_5 = []
    all_mrr = []
    all_top1_hit = []
    
    with torch.no_grad():
        for scene_name, candidates in multi_candidate_scenes.items():
            # 收集该 scene 的所有样本
            scores = []
            relevances = []  # GT relevance (consistency_label)
            
            for cand in candidates:
                idx = cand["index"]
                sample = dataset[idx]
                
                h_imgs = sample["history_images"].unsqueeze(0).to(device)
                f_imgs = sample["future_images"].unsqueeze(0).to(device)
                ego = sample["ego_state"].unsqueeze(0).to(device)
                traj = sample["candidate_traj"].unsqueeze(0).to(device)
                
                out = model(h_imgs, f_imgs, ego, traj)
                score = torch.sigmoid(out["consistency_logit"]).item()
                
                scores.append(score)
                relevances.append(cand["consistency_label"])
            
            # 计算 NDCG@k
            def compute_ndcg(scores_list, relevance_list, k):
                if len(scores_list) < 2:
                    return 0.0
                
                # 按分数排序
                sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
                sorted_relevances = [rel for _, rel in sorted_pairs[:k]]
                
                # DCG
                dcg = sum(
                    rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevances)
                )
                
                # Ideal DCG
                ideal_relevances = sorted(relevance_list, reverse=True)[:k]
                idcg = sum(
                    rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances)
                )
                
                return dcg / idcg if idcg > 0 else 0.0
            
            # 计算 MRR
            def compute_mrr(scores_list, relevance_list):
                if len(scores_list) < 2:
                    return 0.0
                
                # 按分数排序
                sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
                
                # 找到第一个正样本的位置
                for i, (_, rel) in enumerate(sorted_pairs):
                    if rel == 1:
                        return 1.0 / (i + 1)
                return 0.0
            
            # 计算 Top-1 Hit Rate
            def compute_top1_hit(scores_list, relevance_list):
                if len(scores_list) < 2:
                    return 0.0
                
                # 找到分数最高的样本
                best_idx = np.argmax(scores_list)
                return 1.0 if relevance_list[best_idx] == 1 else 0.0
            
            # 累积指标
            all_ndcg_3.append(compute_ndcg(scores, relevances, k=3))
            all_ndcg_5.append(compute_ndcg(scores, relevances, k=5))
            all_mrr.append(compute_mrr(scores, relevances))
            all_top1_hit.append(compute_top1_hit(scores, relevances))
    
    return {
        "ndcg@3": float(np.mean(all_ndcg_3)) if all_ndcg_3 else 0.0,
        "ndcg@5": float(np.mean(all_ndcg_5)) if all_ndcg_5 else 0.0,
        "mrr": float(np.mean(all_mrr)) if all_mrr else 0.0,
        "top1_hit_rate": float(np.mean(all_top1_hit)) if all_top1_hit else 0.0,
        "num_scenes": len(multi_candidate_scenes),
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

    epoch = checkpoint.get("epoch", "?")
    best_val_loss = checkpoint.get("best_val_loss", "?")
    model_type = cfg.get("model_type", "critic")
    print(f"Checkpoint 信息: epoch={epoch}, best_val_loss={best_val_loss}")
    print(f"模型类型: {model_type}")

    # 构建模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "consistency":
        model = ConsistencyCriticModel(cfg).to(device)
    else:
        model = CriticModel(cfg).to(device)
    
    # 兼容旧版模型加载
    model_state = checkpoint["model"]
    current_state = model.state_dict()
    
    has_multi_dim = 'speed_consistency_head.weight' in current_state
    has_old_version = 'speed_consistency_head.weight' not in model_state
    
    if has_multi_dim and has_old_version:
        print("[INFO] 检测到旧版模型，正在兼容加载...")
        matched_keys = set(model_state.keys()) & set(current_state.keys())
        for key in matched_keys:
            current_state[key] = model_state[key]
        
        if 'consistency_head.weight' in model_state:
            missing_keys = set(current_state.keys()) - set(model_state.keys())
            for missing_key in missing_keys:
                if 'head' in missing_key and 'validity' not in missing_key:
                    current_state[missing_key] = model_state[
                        'consistency_head.weight' if 'weight' in missing_key else 'consistency_head.bias'
                    ]
        
        model.load_state_dict(current_state, strict=False)
        print(f"  ✅ 加载 {len(matched_keys)} 个匹配权重")
    else:
        model.load_state_dict(model_state)
    
    print(f"模型加载完成，设备: {device}")

    # 构建数据集
    index_key = "val_index" if args.split == "val" else "train_index"
    index_path = cfg[index_key]
    print(f"数据集: {args.split} ({index_path})")

    if model_type == "consistency":
        dataset = ConsistencyDataset(
            index_path=index_path, cfg=cfg, training=False,
        )
    else:
        dataset = CriticJsonlDataset(
            index_path=index_path, cfg=cfg, training=False,
        )
    print(f"样本总数: {len(dataset)}")

    # 评估
    print("\n开始评估...")

    if model_type == "consistency":
        metrics = evaluate_consistency(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        # 打印 Consistency Critic 评估结果
        print("\n" + "=" * 60)
        print("Consistency Critic 评估结果")
        print("=" * 60)
        print(f"  总样本数: {metrics['total_samples']}")
        _print_head_metrics("Consistency Head", metrics["consistency"])
        _print_head_metrics("Validity Head", metrics["validity"])

        # per-source-type 分组
        if metrics.get("per_source_type"):
            print("\n  [Per Source Type]")
            for st, st_data in metrics["per_source_type"].items():
                print(f"    --- {st} (n={st_data['count']}) ---")
                c = st_data["consistency"]
                v = st_data["validity"]
                print(f"      consistency: {_format_source_line(c)}")
                print(f"      validity:    {_format_source_line(v)}")
        print("=" * 60)
    else:
        metrics = evaluate(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        # 打印原始 Critic 评估结果
        print("\n" + "=" * 50)
        print("评估结果")
        print("=" * 50)
        print(f"  样本数:     {metrics['total_samples']} "
              f"(正: {metrics['num_positive']}, 负: {metrics['num_negative']})")
        print(f"  Loss:       {metrics['avg_loss']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1 Score:   {metrics['f1_score']:.4f}")
        print(f"  混淆矩阵:   TP={metrics['tp']}, FP={metrics['fp']}, "
              f"FN={metrics['fn']}, TN={metrics['tn']}")
        print(f"  正样本平均预测概率: {metrics['pos_prob_mean']:.4f}")
        print(f"  负样本平均预测概率: {metrics['neg_prob_mean']:.4f}")
        print("=" * 50)

    # 保存结果到 JSON
    result_path = ckpt_path.parent.parent / f"eval_{args.split}_results.json"
    
    # 如果启用 ranking 评估
    if args.eval_ranking and model_type == "consistency":
        print("\n" + "=" * 60)
        print("开始 Ranking 评估...")
        print("=" * 60)
        ranking_metrics = compute_ranking_metrics(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
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


if __name__ == "__main__":
    main()
