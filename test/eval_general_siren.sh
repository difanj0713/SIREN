#!/bin/bash

MODEL="qwen3-4b"
DEVICE="cuda"
BATCH_SIZE=64

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
echo "Evaluating SIREN"
echo "========================================"
echo "Model: $MODEL"
echo "Datasets: ${DATASETS[@]}"
echo ""

python evaluate_general_siren.py \
    --model $MODEL \
    --datasets ${DATASETS[@]} \
    --device $DEVICE \
    --batch_size $BATCH_SIZE

echo ""
echo "Done!"
