import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from tools.build_consistency_index import (
    ConsistencyAnchor,
    build_time_shift_future_negatives,
    filter_decodable_anchors,
)
from train import ConsistencyCriticModel


def model_cfg(temporal_encoder: str) -> dict:
    return {
        "baseline_mode": "full",
        "ego_state_dim": 5,
        "candidate_traj_steps": 8,
        "consistency_traj_steps": 4,
        "future_num_frames": 4,
        "traj_dim": 3,
        "model": {
            "image_feature_dim": 32,
            "action_feature_dim": 16,
            "hidden_dim": 32,
            "fusion_dim": 32,
            "dropout": 0.0,
            "use_action_visual_interaction": True,
            "temporal_encoder": temporal_encoder,
        },
    }


def anchor(scene: str, timestamp: int, image: str = "camera/good.jpg") -> ConsistencyAnchor:
    return ConsistencyAnchor(
        sample_id=f"{scene}__{timestamp}",
        scene_name=scene,
        timestamp_us=timestamp,
        history_images=[image] * 4,
        future_images=[image] * 4,
        ego_state=[0.0] * 5,
        candidate_traj=[[float(step), 0.0, 0.0] for step in range(8)],
    )


class TemporalEncodingTests(unittest.TestCase):
    def test_gru_encoder_is_sensitive_to_frame_order(self) -> None:
        torch.manual_seed(0)
        model = ConsistencyCriticModel(model_cfg("gru")).eval()
        images = torch.randn(2, 4, 3, 32, 32)
        with torch.no_grad():
            original = model._encode_images(
                images, model.future_proj, model.future_temporal_encoder,
            )
            reversed_frames = model._encode_images(
                torch.flip(images, dims=[1]),
                model.future_proj,
                model.future_temporal_encoder,
            )
        self.assertFalse(torch.allclose(original, reversed_frames))

    def test_mean_encoder_keeps_legacy_order_invariance(self) -> None:
        torch.manual_seed(0)
        model = ConsistencyCriticModel(model_cfg("mean")).eval()
        images = torch.randn(2, 4, 3, 32, 32)
        with torch.no_grad():
            original = model._encode_images(
                images, model.future_proj, model.future_temporal_encoder,
            )
            reversed_frames = model._encode_images(
                torch.flip(images, dims=[1]),
                model.future_proj,
                model.future_temporal_encoder,
            )
        self.assertTrue(torch.allclose(original, reversed_frames, atol=1e-6))


class IndexBuilderTests(unittest.TestCase):
    def test_time_shift_never_crosses_scene_boundaries(self) -> None:
        anchors = [
            anchor("scene_a", 10),
            anchor("scene_b", 10),
            anchor("scene_a", 20),
            anchor("scene_b", 20),
            anchor("scene_a", 30),
            anchor("scene_b", 30),
        ]
        negatives = build_time_shift_future_negatives(anchors, shift_steps=2)
        self.assertEqual(len(negatives), len(anchors))
        for row in negatives:
            negative_scene = row["negative_source_id"].rsplit("__", 1)[0]
            self.assertEqual(row["scene_name"], negative_scene)

    def test_decode_filter_removes_bad_image_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "camera"
            root.mkdir()
            good = root / "good.jpg"
            bad = root / "bad.jpg"
            Image.new("RGB", (4, 4), color="white").save(good)
            bad.touch()

            scene_anchors = {
                "scene": [
                    anchor("scene", 10, "camera/good.jpg"),
                    anchor("scene", 20, "camera/bad.jpg"),
                ],
            }
            filtered, rejected, examples = filter_decodable_anchors(
                scene_anchors, [root],
            )

        self.assertEqual(rejected, 1)
        self.assertEqual([item.timestamp_us for item in filtered["scene"]], [10])
        self.assertTrue(examples)


if __name__ == "__main__":
    unittest.main()
