#!/bin/bash

MODEL="qwen3-4b"
DEVICE="cuda"

echo "========================================"
echo "SIREN Streaming Evaluation"
echo "========================================"
echo "Model: $MODEL"
echo ""

python evaluate_streaming_siren.py \
    --model $MODEL \
    --splits thinking_loc \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "ERROR: Streaming evaluation failed"
    exit 1
fi

echo ""
echo "Done!"
