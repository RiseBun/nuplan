from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
data_root = Path("/mnt/cpfs/prediction/lipeinan/nuplan_data/mini_set")
index_root = project_root / "indices"


cfg = dict(
    experiment_name="nuplan_consistency_mini_v2",
    model_type="consistency",
    seed=42,
    work_dir=str(project_root / "work_dirs" / "consistency_mini_v2"),
    train_index=str(index_root / "consistency_train.jsonl"),
    val_index=str(index_root / "consistency_val.jsonl"),
    image_root=str(data_root),
    mini_db_root=(
        "/mnt/datasets/e2e-datasets/20260227/e2e-datasets/"
        "dataset_pkgs/nuplan-v1.1/splits/mini"
    ),
    camera_roots=[
        str(data_root / "nuplan-v1.1_mini_camera_0"),
        str(data_root / "nuplan-v1.1_mini_camera_1"),
    ],
    camera_channel="CAM_F0",
    log_interval=20,
    val_interval=1,
    save_interval=1,
    # 训练超参
    epochs=30,
    batch_size=8,
    num_workers=4,
    # 输入规格
    image_size=224,
    history_num_frames=4,
    future_num_frames=4,
    candidate_traj_steps=8,
    future_step_time_s=0.5,
    ego_state_dim=5,
    traj_dim=3,
    # 损失权重 - 多维度评估
    lambda_consistency=1.0,
    lambda_validity=0.5,
    positive_weight=1.0,
    consistency_positive_weight=3.0,  # Consistency Head 正负样本比 1:3
    validity_positive_weight=1.0,     # Validity Head 正负样本比 1:1
    
    # 多维度 consistency 权重（细粒度评估）
    lambda_speed_consistency=0.3,
    lambda_steering_consistency=0.3,
    lambda_progress_consistency=0.2,
    lambda_temporal_coherence=0.2,
    # 优化器
    optimizer=dict(
        lr=1e-4,
        weight_decay=1e-2,
    ),
    # 模型结构
    model=dict(
        image_channels=3,
        image_feature_dim=256,
        action_feature_dim=128,
        hidden_dim=256,
        fusion_dim=256,
        dropout=0.1,
    ),
    # 数据集预处理
    dataset=dict(
        normalize_ego_state=True,
        normalize_candidate_traj=True,
        normalize_mode="linear",          # "linear" 或 "tanh"
        traj_scale=[60.0, 25.0, 2.0],     # dx/dy/dyaw 线性缩放因子
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    ),
    
    # Ranking 评估配置（用于候选排序能力测试）
    ranking=dict(
        enabled=True,
        num_candidates_per_scene=5,       # 每个scene的候选数
        ranking_metrics=["ndcg@3", "ndcg@5", "mrr", "top1_hit_rate"],
    ),
)
