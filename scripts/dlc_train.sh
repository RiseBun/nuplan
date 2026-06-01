#!/bin/bash

# =============================================================================
# NuPlan Consistency Critic DLC Multi-Node Multi-GPU Training Script
# =============================================================================
# Usage:
#   One-command training (auto IAC index build + training):
#     cd /path/to/nuplan && bash scripts/dlc_train.sh
#
#   Custom parameters:
#     bash scripts/dlc_train.sh --epochs=100 --batch-size=32 --work-dir=work_dirs/my_model
#
#   Smoke test (2 epochs, fast validation):
#     bash scripts/dlc_train.sh --smoke-test
#
# Features:
#   - Auto data preparation if index not found
#   - Multi-node multi-GPU distributed training
#   - Automatic GPU detection
#   - NCCL optimization for stable training
# =============================================================================

set -e

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export SETUPTOOLS_USE_DISTUTILS=local
export PYTHONWARNINGS="ignore:UserWarning"
# 使用新版环境变量名（旧版 NCCL_ASYNC_ERROR_HANDLING 已废弃）
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# 增加 NCCL 超时时间（秒），避免多节点通信波动导致误杀
export NCCL_TIMEOUT=1800

###################################
# User Configuration Section
###################################
PYTHON_BIN="${IAC_PYTHON_BIN:-/root/miniconda3/bin/python}"
# Get project root relative to script location (scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="configs/train_consistency_mini.py"
TRAIN_SCRIPT="train.py"

###################################
# DLC Environment Variables (Auto-injected by DLC)
###################################
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# IMPORTANT: Clear DLC's environment variables that conflict with torchrun
# torchrun will set its own WORLD_SIZE, RANK, LOCAL_RANK with correct values
# If not cleared, old values may cause "CUDA error: invalid device ordinal"
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

# Auto-detect GPU count for single-node testing
if [ "$NNODES" -eq 1 ] && [ "$MASTER_ADDR" = "localhost" ]; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$DETECTED_GPUS" -gt 0 ]; then
        NPROC_PER_NODE=$DETECTED_GPUS
    fi
fi

###################################
# Setup Environment
###################################
cd ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python executable not found: $PYTHON_BIN"
    echo "Set IAC_PYTHON_BIN to the Python executable for this environment."
    exit 1
fi

# 提高文件描述符限制，避免多 worker 数据加载时 ancdata 错误
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit to 65536, using default"

# 清理 __pycache__，避免开发机编译的 .pyc 在 DLC 节点上产生路径冲突
find "${PROJECT_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "==================================="
echo "NuPlan Critic DLC Distributed Training"
echo "==================================="
echo "Node Configuration:"
echo "  Total nodes: $NNODES"
echo "  Current node rank: $NODE_RANK"
echo "  GPUs per node: $NPROC_PER_NODE"
echo "  Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "-----------------------------------"
echo "Project Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Config file: $CONFIG_FILE"
echo "  Train script: $TRAIN_SCRIPT"
echo "  Python: $PYTHON_BIN"
echo "==================================="

###################################
# Validate Files
###################################
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

###################################
# Auto Data Preparation (If Needed)
###################################
echo ""
echo "Checking training data..."

INDEX_DIR="${NUPLAN_INDEX_ROOT:-${PROJECT_ROOT}/indices_v3}"
export NUPLAN_INDEX_ROOT="$INDEX_DIR"
TRAIN_INDEX="${INDEX_DIR}/consistency_train.jsonl"
VAL_INDEX="${INDEX_DIR}/consistency_val.jsonl"

if [ ! -f "$TRAIN_INDEX" ] || [ ! -f "$VAL_INDEX" ]; then
    echo "Training index not found!"
    echo "   Train: $TRAIN_INDEX"
    echo "   Val: $VAL_INDEX"
    echo ""
    echo "Building fresh IAC index from nuPlan DB and camera images..."
    echo ""
    
    DATA_ROOT="${NUPLAN_DATA_ROOT:-${PROJECT_ROOT}/..}"
    DB_ROOT="${NUPLAN_DB_ROOT:-${DATA_ROOT}/data/cache/mini}"
    mapfile -t CAMERA_ROOTS < <(
        "$PYTHON_BIN" - <<'PY'
from data_paths import camera_roots
for path in camera_roots():
    print(path)
PY
    )

    if [ ! -d "$DB_ROOT" ]; then
        echo "Error: NuPlan DB root not found: $DB_ROOT"
        echo "Set NUPLAN_DB_ROOT or place DB files under project data/."
        exit 1
    fi

    echo "  DB root: $DB_ROOT"
    echo "  Camera roots: ${CAMERA_ROOTS[*]}"
    echo "  Output: $INDEX_DIR"
    echo ""
    mkdir -p "$INDEX_DIR"
    "$PYTHON_BIN" tools/build_consistency_index.py \
        --db-root "$DB_ROOT" \
        --image-roots "${CAMERA_ROOTS[@]}" \
        --output-dir "$INDEX_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Failed to build IAC index!"
        exit 1
    fi
    
    echo ""
    echo "IAC index build completed!"
    echo ""
else
    echo "Training index found:"
    echo "   Train: $TRAIN_INDEX"
    echo "   Val: $VAL_INDEX"
    echo ""
fi
###################################
# Default Training Arguments
###################################
# Default values (can be overridden by command line args)
DEFAULT_EPOCHS=50
DEFAULT_BATCH_SIZE=16
DEFAULT_WORK_DIR="work_dirs/iac_full"

# Parse arguments (simple key-value parsing)
EPOCHS=$DEFAULT_EPOCHS
BATCH_SIZE=$DEFAULT_BATCH_SIZE
WORK_DIR=$DEFAULT_WORK_DIR
SMOKE_TEST=false

for arg in "$@"; do
    case $arg in
        --epochs=*)
            EPOCHS="${arg#*=}"
            shift
            ;;
        --batch-size=*)
            BATCH_SIZE="${arg#*=}"
            shift
            ;;
        --work-dir=*)
            WORK_DIR="${arg#*=}"
            shift
            ;;
        --smoke-test)
            SMOKE_TEST=true
            EPOCHS=2
            BATCH_SIZE=4
            WORK_DIR="work_dirs/smoke_test"
            shift
            ;;
        *)
            # Pass through other arguments
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# If smoke test, override
if [ "$SMOKE_TEST" = true ]; then
    echo "🔥 SMOKE TEST MODE: 2 epochs, batch_size=4"
    echo ""
fi

###################################
# Build Training Arguments
###################################
EXTRA_ARGS+=("--epochs" "$EPOCHS")
EXTRA_ARGS+=("--batch-size" "$BATCH_SIZE")
EXTRA_ARGS+=("--work-dir" "$WORK_DIR")

echo ""
echo "Extra arguments: ${EXTRA_ARGS[*]}"
echo ""

###################################
# Distributed Training Launch
###################################
echo "Starting distributed training..."

"$PYTHON_BIN" -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $TRAIN_SCRIPT \
    --config $CONFIG_FILE \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Training completed!"
