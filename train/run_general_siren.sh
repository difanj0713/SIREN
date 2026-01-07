#!/bin/bash

MODEL="qwen3-4b"
DEVICE="cuda"
BATCH_SIZE=32
C_VALUES="200.0 500.0 1000.0"
THRESHOLDS="0.9"
N_TRIALS=32
N_JOBS=1
N_FOLDS=5
VAL_RATIO=0.2
USE_GPU_DATA=1
REP_TYPES="residual_mean"

DATASETS=(
    "toxic_chat"
    "openai_moderation"
    "aegis"
    "aegis2"
    "wildguard"
    "safe_rlhf"
    "beavertails"
)

echo "========================================"
echo "Training SIREN"
echo "========================================"
echo "Model: $MODEL"
echo "Datasets: ${DATASETS[@]}"
echo "Device: $DEVICE"
echo ""

python train_general_siren.py \
    --model $MODEL \
    --datasets ${DATASETS[@]} \
    --batch_size $BATCH_SIZE \
    --c_values $C_VALUES \
    --pooling_types $REP_TYPES \
    --thresholds $THRESHOLDS \
    --n_trials $N_TRIALS \
    --n_jobs $N_JOBS \
    --n_folds $N_FOLDS \
    --val_ratio $VAL_RATIO \
    --use_gpu_data $USE_GPU_DATA \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "ERROR: train_general_siren.py failed"
    exit 1
fi

echo ""
echo "Done! Model saved to probes/optuna/${MODEL}_general/"
