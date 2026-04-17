#!/usr/bin/env python3
import argparse
import bisect
import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NuPlan critic JSONL indices")
    parser.add_argument(
        "--db-root",
        default="/mnt/datasets/e2e-datasets/20260227/e2e-datasets/dataset_pkgs/nuplan-v1.1/splits/mini",
        help="Directory with mini split db files",
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            "/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set/nuplan-v1.1_mini_camera_0",
            "/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set/nuplan-v1.1_mini_camera_1",
        ],
        help="One or more extracted camera roots",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/cpfs/prediction/lipeinan/nuplan/indices",
        help="Output directory for JSONL indices",
    )
    parser.add_argument("--camera-channel", default="CAM_F0", help="Camera channel to use")
    parser.add_argument("--history-num-frames", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--future-step-time-s", type=float, default=0.5)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Anchor every N frames to keep the first version compact",
    )
    parser.add_argument(
        "--min-negative-index-gap",
        type=int,
        default=20,
        help="Minimum anchor distance between positive and negative trajectory source",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=0, help="Debug: limit number of scenes")
    parser.add_argument(
        "--max-samples-per-scene",
        type=int,
        default=0,
        help="Debug: limit anchors per scene after sampling",
    )
    return parser.parse_args()


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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
class AnchorSample:
    sample_id: str
    scene_name: str
    timestamp_us: int
    history_images: List[str]
    ego_state: List[float]
    candidate_traj: List[List[float]]


def discover_scene_roots(image_roots: Sequence[Path]) -> Dict[str, Path]:
    scene_to_root: Dict[str, Path] = {}
    for image_root in image_roots:
        if not image_root.exists():
            continue
        for scene_dir in sorted(image_root.iterdir()):
            if scene_dir.is_dir():
                scene_to_root[scene_dir.name] = image_root
    return scene_to_root


def load_scene_anchors(
    db_path: Path,
    scene_name: str,
    image_root: Path,
    camera_channel: str,
    history_num_frames: int,
    future_steps: int,
    future_step_time_s: float,
    sample_stride: int,
    max_samples_per_scene: int,
) -> List[AnchorSample]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            i.timestamp AS image_timestamp,
            i.filename_jpg AS filename_jpg,
            i.ego_pose_token AS ego_pose_token,
            ep.x AS x,
            ep.y AS y,
            ep.qw AS qw,
            ep.qx AS qx,
            ep.qy AS qy,
            ep.qz AS qz,
            ep.vx AS vx,
            ep.vy AS vy,
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

    cur.execute(
        """
        SELECT
            timestamp,
            x,
            y,
            qw,
            qx,
            qy,
            qz,
            vx,
            vy,
            acceleration_x,
            angular_rate_z
        FROM ego_pose
        ORDER BY timestamp
        """
    )
    pose_rows = cur.fetchall()
    conn.close()

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
                    float(row["qw"]),
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                ),
                vx=float(row["vx"]),
                vy=float(row["vy"]),
                acceleration_x=float(row["acceleration_x"]),
                angular_rate_z=float(row["angular_rate_z"]),
            )
        )

    dt_us = int(future_step_time_s * 1e6)
    anchors: List[AnchorSample] = []
    root_prefix = image_root.name

    for image_idx in range(history_num_frames - 1, len(image_rows), sample_stride):
        row = image_rows[image_idx]
        current_pose = EgoPose(
            timestamp=int(row["image_timestamp"]),
            x=float(row["x"]),
            y=float(row["y"]),
            yaw=yaw_from_quaternion(
                float(row["qw"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
            ),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            acceleration_x=float(row["acceleration_x"]),
            angular_rate_z=float(row["angular_rate_z"]),
        )

        history_rows = image_rows[image_idx - history_num_frames + 1 : image_idx + 1]
        history_images = [str(Path(root_prefix) / str(hist["filename_jpg"])) for hist in history_rows]
        if not all((image_root / hist["filename_jpg"]).exists() for hist in history_rows):
            continue

        future_traj: List[List[float]] = []
        valid = True
        cos_yaw = math.cos(-current_pose.yaw)
        sin_yaw = math.sin(-current_pose.yaw)
        for step in range(1, future_steps + 1):
            target_timestamp = current_pose.timestamp + step * dt_us
            pose_idx = bisect.bisect_left(pose_timestamps, target_timestamp)
            if pose_idx >= len(poses):
                valid = False
                break
            pose = poses[pose_idx]
            if abs(pose.timestamp - target_timestamp) > dt_us:
                valid = False
                break
            dx_world = pose.x - current_pose.x
            dy_world = pose.y - current_pose.y
            dx_local = dx_world * cos_yaw - dy_world * sin_yaw
            dy_local = dx_world * sin_yaw + dy_world * cos_yaw
            dyaw = wrap_angle(pose.yaw - current_pose.yaw)
            future_traj.append([dx_local, dy_local, dyaw])

        if not valid:
            continue

        anchors.append(
            AnchorSample(
                sample_id=f"{scene_name}__{current_pose.timestamp}",
                scene_name=scene_name,
                timestamp_us=current_pose.timestamp,
                history_images=history_images,
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

        if max_samples_per_scene > 0 and len(anchors) >= max_samples_per_scene:
            break

    return anchors


def split_scenes(scene_names: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
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


def build_negative_samples(
    anchors: List[AnchorSample],
    min_negative_index_gap: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    negatives: List[Dict] = []
    if len(anchors) < 2:
        return negatives

    for idx, anchor in enumerate(anchors):
        candidate_indices = [
            j for j in range(len(anchors))
            if j != idx and abs(j - idx) >= min_negative_index_gap
        ]
        if not candidate_indices:
            candidate_indices = [j for j in range(len(anchors)) if j != idx]
        neg_idx = rng.choice(candidate_indices)
        neg_source = anchors[neg_idx]
        negatives.append(
            {
                "sample_id": f"{anchor.sample_id}__neg",
                "scene_name": anchor.scene_name,
                "timestamp_us": anchor.timestamp_us,
                "history_images": anchor.history_images,
                "ego_state": anchor.ego_state,
                "candidate_traj": neg_source.candidate_traj,
                "label": 0,
                "negative_source_scene": neg_source.scene_name,
                "negative_source_timestamp_us": neg_source.timestamp_us,
            }
        )
    return negatives


def serialize_split(anchors: List[AnchorSample], seed: int, min_negative_index_gap: int) -> List[Dict]:
    positives = [
        {
            "sample_id": f"{anchor.sample_id}__pos",
            "scene_name": anchor.scene_name,
            "timestamp_us": anchor.timestamp_us,
            "history_images": anchor.history_images,
            "ego_state": anchor.ego_state,
            "candidate_traj": anchor.candidate_traj,
            "label": 1,
        }
        for anchor in anchors
    ]
    negatives = build_negative_samples(anchors, min_negative_index_gap=min_negative_index_gap, seed=seed)
    return positives + negatives


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    all_scene_anchors: Dict[str, List[AnchorSample]] = {}
    skipped_scenes: Dict[str, str] = {}
    for scene_name in scene_names:
        image_root = scene_to_root[scene_name]
        db_path = db_root / f"{scene_name}.db"
        if not db_path.exists():
            skipped_scenes[scene_name] = f"missing db: {db_path}"
            continue
        anchors = load_scene_anchors(
            db_path=db_path,
            scene_name=scene_name,
            image_root=image_root,
            camera_channel=args.camera_channel,
            history_num_frames=args.history_num_frames,
            future_steps=args.future_steps,
            future_step_time_s=args.future_step_time_s,
            sample_stride=args.sample_stride,
            max_samples_per_scene=args.max_samples_per_scene,
        )
        if not anchors:
            skipped_scenes[scene_name] = "no valid anchors"
            continue
        all_scene_anchors[scene_name] = anchors
        print(f"[OK] {scene_name}: anchors={len(anchors)} root={image_root.name}")

    usable_scenes = sorted(all_scene_anchors.keys())
    if not usable_scenes:
        raise RuntimeError("No usable scenes found; please verify mini dbs and extracted camera roots")

    train_scenes, val_scenes = split_scenes(usable_scenes, val_ratio=args.val_ratio, seed=args.seed)
    train_anchors = [anchor for scene in train_scenes for anchor in all_scene_anchors[scene]]
    val_anchors = [anchor for scene in val_scenes for anchor in all_scene_anchors[scene]]

    train_rows = serialize_split(
        train_anchors,
        seed=args.seed,
        min_negative_index_gap=args.min_negative_index_gap,
    )
    val_rows = serialize_split(
        val_anchors,
        seed=args.seed + 1,
        min_negative_index_gap=max(1, args.min_negative_index_gap // 2),
    )

    train_path = output_dir / "critic_train.jsonl"
    val_path = output_dir / "critic_val.jsonl"
    summary_path = output_dir / "critic_index_summary.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    summary = {
        "camera_channel": args.camera_channel,
        "history_num_frames": args.history_num_frames,
        "future_steps": args.future_steps,
        "future_step_time_s": args.future_step_time_s,
        "sample_stride": args.sample_stride,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "num_train_anchors": len(train_anchors),
        "num_val_anchors": len(val_anchors),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "skipped_scenes": skipped_scenes,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote train index: {train_path}")
    print(f"Wrote val index:   {val_path}")
    print(f"Wrote summary:     {summary_path}")


if __name__ == "__main__":
    main()
