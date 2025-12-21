#!/bin/bash

MODEL="qwen3-1.7b"
DATASETS="toxic_chat openai_moderation aegis aegis2 wildguard safe_rlhf beavertails"

EPOCHS=10
BATCH_SIZE=32
PATIENCE=2
LR=1e-3

python train_clf_head_multi_dataset.py \
    --model $MODEL \
    --datasets $DATASETS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE \
    --lr $LR \
    --save_dir clf_heads
