#!/usr/bin/env python3
"""构建 Consistency Critic 训练索引

基于 build_critic_index.py 扩展，增加:
- 未来图像帧提取 (future_images)
- 三种负样本生成: traj_swap / image_swap / perturb
- 双标签: consistency_label, validity_label
- source_type 标识样本来源

用法:
    python tools/build_consistency_index.py
    python tools/build_consistency_index.py --sample-stride 3 --max-scenes 5
"""
import argparse
import bisect
import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 NuPlan Consistency Critic JSONL 索引"
    )
    parser.add_argument(
        "--db-root",
        default="/mnt/datasets/e2e-datasets/20260227/e2e-datasets/"
                "dataset_pkgs/nuplan-v1.1/splits/mini",
        help="mini split db 文件目录",
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            "/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set/"
            "nuplan-v1.1_mini_camera_0",
            "/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set/"
            "nuplan-v1.1_mini_camera_1",
        ],
        help="已解压的 camera 目录",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/cpfs/prediction/lipeinan/nuplan/indices",
        help="输出 JSONL 索引目录",
    )
    parser.add_argument("--camera-channel", default="CAM_F0")
    parser.add_argument("--history-num-frames", type=int, default=4)
    parser.add_argument(
        "--future-image-offsets",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="未来图像时间偏移 (秒)，默认 0.5/1.0/1.5/2.0s 共 4 帧",
    )
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--future-step-time-s", type=float, default=0.5)
    parser.add_argument(
        "--sample-stride", type=int, default=5,
        help="每 N 帧取一个 anchor",
    )
    parser.add_argument("--min-negative-index-gap", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-samples-per-scene", type=int, default=0)
    # 扰动参数
    parser.add_argument(
        "--perturb-lateral-range", type=float, nargs=2,
        default=[0.5, 2.0], help="横向偏移范围 (m)",
    )
    parser.add_argument(
        "--perturb-heading-range", type=float, nargs=2,
        default=[5.0, 15.0], help="航向扰动范围 (度)",
    )
    parser.add_argument(
        "--perturb-speed-range", type=float, nargs=2,
        default=[0.7, 1.3], help="速度缩放范围",
    )
    return parser.parse_args()


# ────────────────── 数学工具 ──────────────────


def yaw_from_quaternion(
    qw: float, qx: float, qy: float, qz: float,
) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ────────────────── 数据结构 ──────────────────


@dataclass
class EgoPose:
    timestamp: int
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    acceleration_x: float
    angular_rate_z: float


@dataclass
class ConsistencyAnchor:
    """Consistency Critic 的 anchor 样本，含历史+未来图像"""
    sample_id: str
    scene_name: str
    timestamp_us: int
    history_images: List[str]
    future_images: List[str]
    ego_state: List[float]
    candidate_traj: List[List[float]]


# ────────────────── 场景发现与加载 ──────────────────


def discover_scene_roots(image_roots: Sequence[Path]) -> Dict[str, Path]:
    """扫描 camera 目录，返回 {scene_name: image_root}"""
    scene_to_root: Dict[str, Path] = {}
    for image_root in image_roots:
        if not image_root.exists():
            continue
        for scene_dir in sorted(image_root.iterdir()):
            if scene_dir.is_dir():
                scene_to_root[scene_dir.name] = image_root
    return scene_to_root


def _find_nearest_image_index(
    image_timestamps: List[int],
    target_ts: int,
    max_diff_us: int = 200_000,
) -> int:
    """二分查找最接近 target_ts 的图像帧索引，超出容差返回 -1"""
    idx = bisect.bisect_left(image_timestamps, target_ts)
    best_idx = -1
    best_diff = max_diff_us + 1
    for candidate in (idx - 1, idx):
        if 0 <= candidate < len(image_timestamps):
            diff = abs(image_timestamps[candidate] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = candidate
    if best_diff > max_diff_us:
        return -1
    return best_idx


def load_scene_anchors(
    db_path: Path,
    scene_name: str,
    image_root: Path,
    camera_channel: str,
    history_num_frames: int,
    future_image_offsets: List[float],
    future_steps: int,
    future_step_time_s: float,
    sample_stride: int,
    max_samples_per_scene: int,
) -> List[ConsistencyAnchor]:
    """加载场景的所有有效 anchor，包含历史图像和未来图像"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 查询所有图像帧及对应 ego_pose
    cur.execute(
        """
        SELECT
            i.timestamp AS image_timestamp,
            i.filename_jpg AS filename_jpg,
            ep.x AS x, ep.y AS y,
            ep.qw AS qw, ep.qx AS qx, ep.qy AS qy, ep.qz AS qz,
            ep.vx AS vx, ep.vy AS vy,
            ep.acceleration_x AS acceleration_x,
            ep.angular_rate_z AS angular_rate_z
        FROM image i
        JOIN camera c ON i.camera_token = c.token
        JOIN ego_pose ep ON i.ego_pose_token = ep.token
        WHERE c.channel = ?
        ORDER BY i.timestamp
        """,
        (camera_channel,),
    )
    image_rows = cur.fetchall()
    if len(image_rows) < history_num_frames:
        conn.close()
        return []

    # 完整 ego_pose 序列（用于未来轨迹插值）
    cur.execute(
        """
        SELECT timestamp, x, y, qw, qx, qy, qz, vx, vy,
               acceleration_x, angular_rate_z
        FROM ego_pose ORDER BY timestamp
        """
    )
    pose_rows = cur.fetchall()
    conn.close()

    # 构建 ego_pose 时间序列
    pose_timestamps: List[int] = []
    poses: List[EgoPose] = []
    for row in pose_rows:
        pose_timestamps.append(int(row["timestamp"]))
        poses.append(
            EgoPose(
                timestamp=int(row["timestamp"]),
                x=float(row["x"]),
                y=float(row["y"]),
                yaw=yaw_from_quaternion(
                    float(row["qw"]), float(row["qx"]),
                    float(row["qy"]), float(row["qz"]),
                ),
                vx=float(row["vx"]),
                vy=float(row["vy"]),
                acceleration_x=float(row["acceleration_x"]),
                angular_rate_z=float(row["angular_rate_z"]),
            )
        )

    # 图像时间戳索引
    image_timestamps = [int(r["image_timestamp"]) for r in image_rows]

    dt_us = int(future_step_time_s * 1e6)
    anchors: List[ConsistencyAnchor] = []
    root_prefix = image_root.name

    for img_idx in range(
        history_num_frames - 1, len(image_rows), sample_stride,
    ):
        row = image_rows[img_idx]
        current_ts = int(row["image_timestamp"])
        current_pose = EgoPose(
            timestamp=current_ts,
            x=float(row["x"]),
            y=float(row["y"]),
            yaw=yaw_from_quaternion(
                float(row["qw"]), float(row["qx"]),
                float(row["qy"]), float(row["qz"]),
            ),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            acceleration_x=float(row["acceleration_x"]),
            angular_rate_z=float(row["angular_rate_z"]),
        )

        # ---- 历史图像 ----
        hist_rows = image_rows[img_idx - history_num_frames + 1: img_idx + 1]
        history_images = [
            str(Path(root_prefix) / str(h["filename_jpg"]))
            for h in hist_rows
        ]
        if not all(
            (image_root / h["filename_jpg"]).exists() for h in hist_rows
        ):
            continue

        # ---- 未来图像 ----
        future_images: List[str] = []
        future_ok = True
        for offset_s in future_image_offsets:
            target_ts = current_ts + int(offset_s * 1e6)
            fi_idx = _find_nearest_image_index(image_timestamps, target_ts)
            if fi_idx < 0:
                future_ok = False
                break
            fi_row = image_rows[fi_idx]
            if not (image_root / fi_row["filename_jpg"]).exists():
                future_ok = False
                break
            future_images.append(
                str(Path(root_prefix) / str(fi_row["filename_jpg"]))
            )
        if not future_ok:
            continue

        # ---- 未来轨迹（ego 坐标系） ----
        future_traj: List[List[float]] = []
        traj_ok = True
        cos_yaw = math.cos(-current_pose.yaw)
        sin_yaw = math.sin(-current_pose.yaw)
        for step in range(1, future_steps + 1):
            target_ts = current_pose.timestamp + step * dt_us
            p_idx = bisect.bisect_left(pose_timestamps, target_ts)
            if p_idx >= len(poses):
                traj_ok = False
                break
            pose = poses[p_idx]
            if abs(pose.timestamp - target_ts) > dt_us:
                traj_ok = False
                break
            dx_w = pose.x - current_pose.x
            dy_w = pose.y - current_pose.y
            dx_l = dx_w * cos_yaw - dy_w * sin_yaw
            dy_l = dx_w * sin_yaw + dy_w * cos_yaw
            dyaw = wrap_angle(pose.yaw - current_pose.yaw)
            future_traj.append([dx_l, dy_l, dyaw])

        if not traj_ok:
            continue

        anchors.append(
            ConsistencyAnchor(
                sample_id=f"{scene_name}__{current_pose.timestamp}",
                scene_name=scene_name,
                timestamp_us=current_pose.timestamp,
                history_images=history_images,
                future_images=future_images,
                ego_state=[
                    current_pose.vx,
                    current_pose.vy,
                    current_pose.yaw,
                    current_pose.acceleration_x,
                    current_pose.angular_rate_z,
                ],
                candidate_traj=future_traj,
            )
        )

        if 0 < max_samples_per_scene <= len(anchors):
            break

    return anchors


def split_scenes(
    scene_names: List[str], val_ratio: float, seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    names = list(scene_names)
    rng.shuffle(names)
    if len(names) == 1:
        return names, names
    val_count = max(1, int(round(len(names) * val_ratio)))
    val_scenes = sorted(names[:val_count])
    train_scenes = sorted(names[val_count:])
    if not train_scenes:
        train_scenes = val_scenes
    return train_scenes, val_scenes


# ────────────────── 负样本生成 ──────────────────


def _pick_remote_index(
    idx: int,
    total: int,
    min_gap: int,
    rng: random.Random,
) -> int:
    """选择与 idx 距离 >= min_gap 的随机索引"""
    candidates = [
        j for j in range(total)
        if j != idx and abs(j - idx) >= min_gap
    ]
    if not candidates:
        candidates = [j for j in range(total) if j != idx]
    return rng.choice(candidates)


def build_traj_swap_negatives(
    anchors: List[ConsistencyAnchor],
    min_gap: int,
    rng: random.Random,
) -> List[Dict]:
    """N1: 替换轨迹为其他 anchor 的轨迹

    consistency=0 (轨迹与图像不匹配), validity=0 (轨迹与当前上下文不匹配)
    """
    negatives: List[Dict] = []
    n = len(anchors)
    if n < 2:
        return negatives
    for idx, anchor in enumerate(anchors):
        neg_idx = _pick_remote_index(idx, n, min_gap, rng)
        neg_src = anchors[neg_idx]
        negatives.append({
            "sample_id": f"{anchor.sample_id}__traj_swap",
            "scene_name": anchor.scene_name,
            "timestamp_us": anchor.timestamp_us,
            "history_images": anchor.history_images,
            "future_images": anchor.future_images,
            "ego_state": anchor.ego_state,
            "candidate_traj": neg_src.candidate_traj,
            "consistency_label": 0,
            "validity_label": 0,
            "source_type": "traj_swap",
        })
    return negatives


def build_image_swap_negatives(
    anchors: List[ConsistencyAnchor],
    min_gap: int,
    rng: random.Random,
) -> List[Dict]:
    """N2: 替换未来图像为其他 anchor 的未来图像

    consistency=0 (图像与轨迹不匹配), validity=1 (图像本身是真实合理的)
    """
    negatives: List[Dict] = []
    n = len(anchors)
    if n < 2:
        return negatives
    for idx, anchor in enumerate(anchors):
        neg_idx = _pick_remote_index(idx, n, min_gap, rng)
        neg_src = anchors[neg_idx]
        negatives.append({
            "sample_id": f"{anchor.sample_id}__image_swap",
            "scene_name": anchor.scene_name,
            "timestamp_us": anchor.timestamp_us,
            "history_images": anchor.history_images,
            "future_images": neg_src.future_images,
            "ego_state": anchor.ego_state,
            "candidate_traj": anchor.candidate_traj,
            "consistency_label": 0,
            "validity_label": 1,
            "source_type": "image_swap",
        })
    return negatives


def perturb_trajectory(
    traj: List[List[float]],
    perturb_type: str,
    rng: random.Random,
    lateral_range: Tuple[float, float] = (0.5, 2.0),
    heading_range: Tuple[float, float] = (5.0, 15.0),
    speed_range: Tuple[float, float] = (0.7, 1.3),
) -> List[List[float]]:
    """对 GT 轨迹施加语义扰动，生成 "很像但不匹配" 的轨迹"""
    perturbed = [list(pt) for pt in traj]
    n_steps = len(perturbed)

    if perturb_type == "lateral":
        # 横向偏移: 逐步线性增大，模拟偏航漂移
        offset = rng.uniform(*lateral_range) * rng.choice([-1, 1])
        for i, pt in enumerate(perturbed):
            ratio = (i + 1) / n_steps
            pt[1] += offset * ratio

    elif perturb_type == "heading":
        # 航向扰动: 叠加逐步增大的 yaw 偏移
        delta_deg = rng.uniform(*heading_range) * rng.choice([-1, 1])
        delta_rad = math.radians(delta_deg)
        for i, pt in enumerate(perturbed):
            ratio = (i + 1) / n_steps
            d_yaw = delta_rad * ratio
            pt[2] = wrap_angle(pt[2] + d_yaw)
            # 航向变化带来位移偏移
            fwd = abs(pt[0]) if abs(pt[0]) > 0.01 else 0.5
            pt[0] += math.sin(d_yaw) * fwd * 0.1
            pt[1] += (1.0 - math.cos(d_yaw)) * fwd * 0.1

    elif perturb_type == "speed":
        # 速度缩放: 缩放纵向+横向位移
        scale = rng.uniform(*speed_range)
        while 0.9 < scale < 1.1:
            scale = rng.uniform(*speed_range)
        for pt in perturbed:
            pt[0] *= scale
            pt[1] *= scale

    return perturbed


def build_perturb_negatives(
    anchors: List[ConsistencyAnchor],
    rng: random.Random,
    lateral_range: Tuple[float, float],
    heading_range: Tuple[float, float],
    speed_range: Tuple[float, float],
) -> List[Dict]:
    """N3: 对 GT 轨迹做语义扰动

    consistency=0 (轨迹被修改，不再匹配真实未来), validity=0
    """
    perturb_types = ["lateral", "heading", "speed"]
    negatives: List[Dict] = []
    for anchor in anchors:
        ptype = rng.choice(perturb_types)
        new_traj = perturb_trajectory(
            anchor.candidate_traj,
            perturb_type=ptype,
            rng=rng,
            lateral_range=lateral_range,
            heading_range=heading_range,
            speed_range=speed_range,
        )
        negatives.append({
            "sample_id": f"{anchor.sample_id}__perturb_{ptype}",
            "scene_name": anchor.scene_name,
            "timestamp_us": anchor.timestamp_us,
            "history_images": anchor.history_images,
            "future_images": anchor.future_images,
            "ego_state": anchor.ego_state,
            "candidate_traj": new_traj,
            "consistency_label": 0,
            "validity_label": 0,
            "source_type": f"perturb_{ptype}",
        })
    return negatives


# ────────────────── 序列化与输出 ──────────────────


def serialize_split(
    anchors: List[ConsistencyAnchor],
    seed: int,
    min_gap: int,
    lateral_range: Tuple[float, float],
    heading_range: Tuple[float, float],
    speed_range: Tuple[float, float],
) -> List[Dict]:
    """将 anchor 列表转为正样本 + 三类负样本 (正:负 = 1:3)"""
    rng = random.Random(seed)

    # 正样本: gt_pos
    positives = [
        {
            "sample_id": f"{a.sample_id}__gt_pos",
            "scene_name": a.scene_name,
            "timestamp_us": a.timestamp_us,
            "history_images": a.history_images,
            "future_images": a.future_images,
            "ego_state": a.ego_state,
            "candidate_traj": a.candidate_traj,
            "consistency_label": 1,
            "validity_label": 1,
            "source_type": "gt_pos",
        }
        for a in anchors
    ]

    # 三类负样本，每类 1 个/正样本
    neg_traj = build_traj_swap_negatives(anchors, min_gap, rng)
    neg_img = build_image_swap_negatives(anchors, min_gap, rng)
    neg_perturb = build_perturb_negatives(
        anchors, rng, lateral_range, heading_range, speed_range,
    )

    all_rows = positives + neg_traj + neg_img + neg_perturb
    rng.shuffle(all_rows)
    return all_rows


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_traj_scale_factors(
    anchors: List[ConsistencyAnchor],
    traj_dim: int = 3,
) -> List[float]:
    """统计轨迹各维度的最大绝对值，用于线性缩放归一化

    Args:
        anchors: anchor 列表
        traj_dim: 轨迹维度数（默认 3: dx, dy, dyaw）

    Returns:
        各维度的缩放因子，向上取整并设下限为 1.0
    """
    dim_max = [0.0] * traj_dim
    for anchor in anchors:
        for pt in anchor.candidate_traj:
            for d in range(min(len(pt), traj_dim)):
                dim_max[d] = max(dim_max[d], abs(pt[d]))
    # 向上取整并设下限，避免除以零
    return [float(max(math.ceil(v), 1)) for v in dim_max]


def count_source_types(rows: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        st = r.get("source_type", "unknown")
        counts[st] = counts.get(st, 0) + 1
    return counts


# ────────────────── 主流程 ──────────────────


def main() -> None:
    args = parse_args()
    image_roots = [Path(p) for p in args.image_roots]
    db_root = Path(args.db_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_to_root = discover_scene_roots(image_roots)
    scene_names = sorted(scene_to_root.keys())
    if args.max_scenes > 0:
        scene_names = scene_names[: args.max_scenes]
    print(f"发现 {len(scene_names)} 个场景")

    all_scene_anchors: Dict[str, List[ConsistencyAnchor]] = {}
    skipped_scenes: Dict[str, str] = {}

    for scene_name in scene_names:
        image_root = scene_to_root[scene_name]
        db_path = db_root / f"{scene_name}.db"
        if not db_path.exists():
            skipped_scenes[scene_name] = f"缺少 db: {db_path}"
            continue
        anchors = load_scene_anchors(
            db_path=db_path,
            scene_name=scene_name,
            image_root=image_root,
            camera_channel=args.camera_channel,
            history_num_frames=args.history_num_frames,
            future_image_offsets=args.future_image_offsets,
            future_steps=args.future_steps,
            future_step_time_s=args.future_step_time_s,
            sample_stride=args.sample_stride,
            max_samples_per_scene=args.max_samples_per_scene,
        )
        if not anchors:
            skipped_scenes[scene_name] = "无有效 anchor"
            continue
        all_scene_anchors[scene_name] = anchors
        print(
            f"[OK] {scene_name}: anchors={len(anchors)}"
            f" root={image_root.name}"
        )

    usable_scenes = sorted(all_scene_anchors.keys())
    if not usable_scenes:
        raise RuntimeError(
            "未找到可用场景，请检查 mini db 和 camera 目录"
        )

    train_scenes, val_scenes = split_scenes(
        usable_scenes, val_ratio=args.val_ratio, seed=args.seed,
    )
    train_anchors = [
        a for s in train_scenes for a in all_scene_anchors[s]
    ]
    val_anchors = [
        a for s in val_scenes for a in all_scene_anchors[s]
    ]

    # 统计训练集轨迹各维度的缩放因子
    traj_scale = compute_traj_scale_factors(train_anchors)
    print(f"训练集 traj_scale_factors: {traj_scale}")

    lat_range = tuple(args.perturb_lateral_range)
    hdg_range = tuple(args.perturb_heading_range)
    spd_range = tuple(args.perturb_speed_range)

    train_rows = serialize_split(
        train_anchors,
        seed=args.seed,
        min_gap=args.min_negative_index_gap,
        lateral_range=lat_range,
        heading_range=hdg_range,
        speed_range=spd_range,
    )
    val_rows = serialize_split(
        val_anchors,
        seed=args.seed + 1,
        min_gap=max(1, args.min_negative_index_gap // 2),
        lateral_range=lat_range,
        heading_range=hdg_range,
        speed_range=spd_range,
    )

    train_path = output_dir / "consistency_train.jsonl"
    val_path = output_dir / "consistency_val.jsonl"
    summary_path = output_dir / "consistency_index_summary.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    summary = {
        "camera_channel": args.camera_channel,
        "history_num_frames": args.history_num_frames,
        "future_image_offsets": args.future_image_offsets,
        "future_steps": args.future_steps,
        "future_step_time_s": args.future_step_time_s,
        "sample_stride": args.sample_stride,
        "perturb_lateral_range": list(args.perturb_lateral_range),
        "perturb_heading_range": list(args.perturb_heading_range),
        "perturb_speed_range": list(args.perturb_speed_range),
        "traj_scale_factors": traj_scale,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "num_train_anchors": len(train_anchors),
        "num_val_anchors": len(val_anchors),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "train_source_types": count_source_types(train_rows),
        "val_source_types": count_source_types(val_rows),
        "skipped_scenes": skipped_scenes,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n写入训练索引: {train_path} ({len(train_rows)} 条)")
    print(f"写入验证索引: {val_path} ({len(val_rows)} 条)")
    print(f"写入摘要:     {summary_path}")
    print(f"\n训练集样本分布: {count_source_types(train_rows)}")
    print(f"验证集样本分布: {count_source_types(val_rows)}")


if __name__ == "__main__":
    main()
