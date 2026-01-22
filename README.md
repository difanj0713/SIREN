# SIREN: Leveraging Internal Representations for LLM Safeguard

Official repository for our paper ***SIREN: Leveraging Internal Representations for LLM Safeguard***. A demo of our trained model in deployment:

![Demo2](https://github.com/user-attachments/assets/0856b8d5-2547-4520-8869-4722391c1a60)


## Setup

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate siren
```

## Training

Train SIREN on multiple safety datasets with a specific backbone model:

```bash
cd train
bash run_general_siren.sh
```

The training script will:
- Extract internal representations from the model on training sets
- Train layer-wise probes with L1 regularization
- Aggregate safety neurons selected by probes and train the MLP classifier atop
- Save the best model to `train/probes/optuna/{MODEL}_general/best_model.pkl`

## Evaluation

Evaluate the trained SIREN model on test sets:

```bash
cd test
bash eval_general_siren.sh
```

The evaluation script will:
- Load the trained model from `train/probes/optuna/{MODEL}_general/best_model.pkl`
- Extract representations and run inference on test datasets
- Save results to `train/probes/optuna/{MODEL}_general/eval_results.json`

Make sure the `MODEL` variable in `eval_general_siren.sh` matches the model you trained.

## Main Results

Performance comparison (Macro F1) of SIREN against safety-specialized models on existing safety benchmarks:

| Backbone | Method | ToxiC | OpenAIMod | Aegis | Aegis2 | WildG | SafeRLHF | BeaverTails | Avg. |
|----------|--------|-------|-----------|-------|--------|-------|----------|-------------|------|
| Qwen3-0.6B | SIREN | 81.4 | **90.0** | **83.3** | **82.0** | 86.1 | **91.5** | **82.9** | **85.3** |
| Qwen3-0.6B | Guard | **82.0** | 75.9 | 78.8 | 82.0 | **89.1** | 86.9 | 77.1 | 81.7 |
| Llama3.2-1B | SIREN | **80.0** | **92.9** | **82.1** | **82.7** | **86.5** | **92.0** | **83.7** | **85.7** |
| Llama3.2-1B | Guard | 63.3 | 67.5 | 59.5 | 72.6 | 78.6 | 83.3 | 70.0 | 70.7 |
| Qwen3-4B | SIREN | 83.5 | **91.2** | **82.9** | 83.4 | 88.3 | **93.2** | **84.3** | **86.7** |
| Qwen3-4B | Guard | **84.9** | 78.3 | 78.2 | **82.5** | **90.6** | 89.2 | 80.1 | 83.4 |
| Llama3.1-8B | SIREN | **83.1** | **92.0** | **82.9** | **82.9** | **86.7** | **92.5** | **83.8** | **86.3** |
| Llama3.1-8B | Guard | 72.2 | 85.3 | 67.1 | 78.0 | 81.3 | 86.2 | 68.8 | 77.0 |
