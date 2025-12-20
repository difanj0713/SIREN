import sys
sys.path.append('../spin')

import argparse
import pickle
import json
import torch

from config import POOLING_STRATEGIES, REPRESENTATION_TYPES
from probe_trainer import train_and_evaluate_probe, extract_layer_features

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-0.6b")
parser.add_argument("--dataset", type=str, default="toxic_chat")
parser.add_argument("--c_values", type=float, nargs="+", default=[1.0, 5.0, 10.0, 50.0, 100.0])
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--rep_types", type=str, nargs="+", default=["residual_mean", "mlp_mean"])
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
print(f"C values: {args.c_values}")
print(f"Device: {device}")

with open(f"representations/{args.model}_{args.dataset}_train.pkl", "rb") as f:
    train_data = pickle.load(f)
with open(f"representations/{args.model}_{args.dataset}_validation.pkl", "rb") as f:
    val_data = pickle.load(f)
with open(f"representations/{args.model}_{args.dataset}_test.pkl", "rb") as f:
    test_data = pickle.load(f)

train_reps = train_data["representations"]
train_labels = train_data["labels"]
val_reps = val_data["representations"]
val_labels = val_data["labels"]
test_reps = test_data["representations"]
test_labels = test_data["labels"]
num_layers = train_data["num_layers"]

print(f"\nTrain samples: {len(train_labels)}")
print(f"Validation samples: {len(val_labels)}")
print(f"Test samples: {len(test_labels)}")
print(f"Number of layers: {num_layers}")

best_probes = {}
all_results = []

pooling_types = args.rep_types

for pooling_type in pooling_types:
    print(f"\nTraining {pooling_type} probes...")

    for layer_idx in range(num_layers):
        rep_type, pooling = pooling_type.split("_")[0], "_".join(pooling_type.split("_")[1:])

        probe, train_f1, val_f1, best_C = train_and_evaluate_probe(
            train_reps, train_labels,
            val_reps, val_labels,
            layer_idx, rep_type, pooling, args.c_values, device, metric='f1_macro'
        )

        test_X = extract_layer_features(test_reps, layer_idx, rep_type, pooling)
        test_f1 = probe.evaluate(test_X, test_labels, metric='f1_macro')
        test_acc = probe.evaluate(test_X, test_labels, metric='accuracy')

        key = f"layer{layer_idx}_{pooling_type}"
        best_probes[key] = {
            "layer": layer_idx,
            "rep_type": rep_type,
            "pooling": pooling,
            "best_C": best_C,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "probe": probe
        }

        all_results.append({
            "layer": layer_idx,
            "rep_type": rep_type,
            "pooling": pooling,
            "C": best_C,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "test_acc": test_acc
        })

        print(f"  Layer {layer_idx:2d}: Train_F1={train_f1:.4f} Val_F1={val_f1:.4f} Test_F1={test_f1:.4f} Test_Acc={test_acc:.4f} C={best_C}")

output = {
    "best_probes": best_probes,
    "all_results": all_results,
    "model": args.model,
    "dataset": args.dataset
}

output_file = f"probes/{args.model}_{args.dataset}_probes.pkl"
with open(output_file, "wb") as f:
    pickle.dump(output, f)

print(f"\nSaved probes to {output_file}")

print("\nTop 10 best probes by validation F1:")
sorted_probes = sorted(best_probes.items(), key=lambda x: x[1]["val_f1"], reverse=True)
for i, (key, info) in enumerate(sorted_probes[:10]):
    print(f"{i+1}. {key}: val_f1={info['val_f1']:.4f}, test_f1={info['test_f1']:.4f}, C={info['best_C']}")

results_file = f"probes/{args.model}_{args.dataset}_results.json"
json_results = {
    "top_probes": [
        {
            "key": key,
            "layer": info["layer"],
            "rep_type": info["rep_type"],
            "pooling": info["pooling"],
            "val_f1": info["val_f1"],
            "test_f1": info["test_f1"],
            "test_acc": info["test_acc"],
            "train_f1": info["train_f1"],
            "best_C": info["best_C"]
        }
        for key, info in sorted_probes[:20]
    ]
}
with open(results_file, "w") as f:
    json.dump(json_results, f, indent=2)

print(f"Saved results summary to {results_file}")
