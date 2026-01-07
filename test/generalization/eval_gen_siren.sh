#!/bin/bash

MODELS=(
    # "qwen3-0.6b"
    "qwen3-4b"
)

DEVICE="cuda"
BATCH_SIZE=64

echo "========================================"
echo "Evaluating SIREN Generalization on Qwen3GuardTest"
echo "========================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Model: $MODEL"
    echo "========================================"

    python evaluate_think_siren.py \
        --model $MODEL \
        --device $DEVICE \
        --batch_size $BATCH_SIZE
done
