#!/bin/bash

MODEL="qwen3guard-0.6b"
DEVICE="cuda"
BATCH_SIZE=128
C_VALUES="50.0 100.0"
THRESHOLDS="0.6 0.8 0.9"
N_TRIALS=16

# For reference, we provide an empirical mapping from dataset to representation type for better effiency; 
# For a complete analysis, run with "residual_mean mlp_mean residual_last mlp_last" for all datasets.
declare -A DATASET_REP_TYPES
DATASET_REP_TYPES["toxic_chat"]="residual_mean mlp_mean"
DATASET_REP_TYPES["openai_moderation"]="residual_mean mlp_mean"
DATASET_REP_TYPES["aegis"]="residual_mean mlp_mean"
DATASET_REP_TYPES["aegis2"]="residual_mean mlp_mean"
DATASET_REP_TYPES["wildguard"]="residual_mean"
DATASET_REP_TYPES["safe_rlhf"]="residual_mean"
DATASET_REP_TYPES["beavertails"]="residual_mean"

DATASETS=(
    "toxic_chat"
    "openai_moderation"
    "aegis"
    "aegis2"
    "wildguard"
    "safe_rlhf"
    "beavertails"
)

echo "=========================================="
echo "SPIN Pipeline for Model: $MODEL"
echo "Datasets: ${DATASETS[@]}"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing Dataset: $DATASET"
    echo "=========================================="

    REP_TYPES=${DATASET_REP_TYPES[$DATASET]}
    echo "Representation types: $REP_TYPES"

    TRAIN_REP="representations/${MODEL}_${DATASET}_train.pkl"
    VAL_REP="representations/${MODEL}_${DATASET}_validation.pkl"
    TEST_REP="representations/${MODEL}_${DATASET}_test.pkl"
    PROBE_FILE="probes/${MODEL}_${DATASET}_probes.pkl"

    if [ -f "$TRAIN_REP" ] && [ -f "$VAL_REP" ] && [ -f "$TEST_REP" ]; then
        echo "[1/4] Representations already exist, skipping extraction..."
    else
        echo "[1/4] Extracting representations..."
        python extract_representations.py \
            --model $MODEL \
            --dataset $DATASET \
            --batch_size $BATCH_SIZE \
            --device $DEVICE \
            --rep_types $REP_TYPES

        if [ $? -ne 0 ]; then
            echo "ERROR: extract_representations.py failed for $DATASET"
            continue
        fi
    fi

    if [ -f "$PROBE_FILE" ]; then
        echo "[2/4] Probes already exist, skipping training..."
    else
        echo "[2/4] Training sparse probes..."
        python sparsify.py \
            --model $MODEL \
            --dataset $DATASET \
            --c_values $C_VALUES \
            --device $DEVICE \
            --rep_types $REP_TYPES

        if [ $? -ne 0 ]; then
            echo "ERROR: sparsify.py failed for $DATASET"
            continue
        fi
    fi

    echo "[3/4] Training final MLP with neuron aggregation..."
    python aggregate_neurons.py \
        --model $MODEL \
        --dataset $DATASET \
        --pooling_types $REP_TYPES \
        --thresholds $THRESHOLDS \
        --n_trials $N_TRIALS \
        --device $DEVICE

    if [ $? -ne 0 ]; then
        echo "ERROR: aggregate_neurons.py failed for $DATASET"
        continue
    fi

    echo "[4/4] Cleaning up representations..."
    rm -f representations/${MODEL}_${DATASET}_train.pkl
    rm -f representations/${MODEL}_${DATASET}_validation.pkl
    rm -f representations/${MODEL}_${DATASET}_test.pkl
    echo "Removed representations for $DATASET"

    echo "✓ Completed $DATASET"
    echo ""
done

echo "Done!"
