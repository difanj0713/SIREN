import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-0.6b")
parser.add_argument("--dataset", type=str, default="tweet_hate")
parser.add_argument("--pooling_type", type=str, default="residual_mean")
args = parser.parse_args()

probes_file = f"../probes/{args.model}_{args.dataset}_probes.pkl"
with open(probes_file, "rb") as f:
    probe_data = pickle.load(f)

best_probes = probe_data["best_probes"]

layers_data = []
for layer_idx in range(28):
    key = f"layer{layer_idx}_{args.pooling_type}"
    if key in best_probes:
        probe = best_probes[key]["probe"]
        weights = probe.get_feature_importance()
        layers_data.append({
            "layer": layer_idx,
            "weights": weights,
            "val_acc": best_probes[key]["val_acc"],
            "C": best_probes[key]["best_C"]
        })

fig, axes = plt.subplots(7, 4, figsize=(16, 20))
fig.suptitle(f'Probe Weight Sparsity: {args.pooling_type}\n{args.model} on {args.dataset}', fontsize=16)

for i, data in enumerate(layers_data):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    
    weights = data["weights"]
    weights_sorted = np.sort(np.abs(weights))[::-1]
    
    ax.hist(weights, bins=50, alpha=0.7, color='blue', density=True)
    ax.set_title(f'Layer {data["layer"]}\nVal: {data["val_acc"]:.3f}, C: {data["C"]}', fontsize=10)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    
    top_10_pct = int(0.1 * len(weights))
    top_10_weight_sum = np.sum(weights_sorted[:top_10_pct])
    total_weight_sum = np.sum(weights_sorted)
    sparsity_ratio = top_10_weight_sum / total_weight_sum if total_weight_sum > 0 else 0
    
    ax.text(0.05, 0.95, f'Top 10%: {sparsity_ratio:.2f}', 
            transform=ax.transAxes, fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

for i in range(len(layers_data), 28):
    row = i // 4
    col = i % 4
    axes[row, col].set_visible(False)

plt.tight_layout()
plt.savefig(f'{args.pooling_type}_sparsity_{args.model}_{args.dataset}.png', dpi=300, bbox_inches='tight')
print(f"Saved sparsity plot to {args.pooling_type}_sparsity_{args.model}_{args.dataset}.png")

print(f"\nSparsity Analysis for {args.pooling_type}:")
print("Layer | Val Acc | C Value | Top 10% Weight Ratio | Mean Weight | Std Weight")
print("-" * 75)
for data in layers_data:
    weights = data["weights"]
    weights_sorted = np.sort(np.abs(weights))[::-1]
    top_10_pct = int(0.1 * len(weights))
    top_10_weight_sum = np.sum(weights_sorted[:top_10_pct])
    total_weight_sum = np.sum(weights_sorted)
    sparsity_ratio = top_10_weight_sum / total_weight_sum if total_weight_sum > 0 else 0
    
    print(f"{data['layer']:5d} | {data['val_acc']:7.3f} | {data['C']:7.1f} | {sparsity_ratio:18.3f} | {np.mean(np.abs(weights)):10.6f} | {np.std(weights):10.6f}")