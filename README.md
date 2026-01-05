# SPIN Refactored

[![Demo Video](Demo_thumbnail.png)](Demo.mp4)


## Pipeline Overview

1. **Extract Internal Representations**: Extract hidden states from transformer layers with 4 pooling strategies
2. **Train Layer-wise Probes**: Train L1-regularized linear probes on each layer independently  
3. **Aggregate Salient Neurons**: Select top neurons and train meta-classifier on aggregated features
4. **LLM Inference**: Evaluate baseline and SPIN-augmented reasoning with vLLM + LLM-as-judge

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Extract representations (4 pooling methods: residual_mean, residual_last, attn_mean, attn_last)
cd spin/
python extract_representations.py --model qwen3-0.6b --dataset tweet_hate

# 2. Train layer-wise probes with L1 regularization
python sparsify.py --model qwen3-0.6b --dataset tweet_hate

# 3. Aggregate neurons and train meta-classifier
python aggregate_neurons.py --model qwen3-0.6b --dataset tweet_hate --threshold 0.6

# 4. Run SPIN inference && verbalization
python spin_inference.py --model qwen3-0.6b --dataset tweet_hate
```

## A minimal run on TweetEval-Hate dataset

**Base-CoT: 0.4215**

**SPIN-CLF: 0.5980**

**SPIN-CoT: 0.5785**
