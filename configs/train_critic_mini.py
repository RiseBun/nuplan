from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
data_root = Path("/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set")
index_root = project_root / "indices"


cfg = dict(
    experiment_name="nuplan_critic_mini_v1",
    seed=42,
    work_dir=str(project_root / "work_dirs" / "critic_mini_v1"),
    train_index=str(index_root / "critic_train.jsonl"),
    val_index=str(index_root / "critic_val.jsonl"),
    image_root=str(data_root),
    mini_db_root="/mnt/datasets/e2e-datasets/20260227/e2e-datasets/dataset_pkgs/nuplan-v1.1/splits/mini",
    camera_roots=[
        str(data_root / "nuplan-v1.1_mini_camera_0"),
        str(data_root / "nuplan-v1.1_mini_camera_1"),
    ],
    camera_channel="CAM_F0",
    log_interval=20,
    val_interval=1,
    save_interval=1,
    epochs=10,
    batch_size=8,
    num_workers=4,
    image_size=224,
    history_num_frames=4,
    candidate_traj_steps=8,
    future_step_time_s=0.5,
    ego_state_dim=5,
    traj_dim=3,
    positive_weight=1.0,
    optimizer=dict(
        lr=1e-4,
        weight_decay=1e-2,
    ),
    model=dict(
        image_channels=3,
        image_feature_dim=256,
        action_feature_dim=128,
        hidden_dim=256,
        dropout=0.1,
    ),
    dataset=dict(
        normalize_ego_state=True,
        normalize_candidate_traj=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    ),
)
