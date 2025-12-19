import sys
sys.path.append('../spin')

import argparse
import pickle
import json
import numpy as np
import torch

from config import POOLING_STRATEGIES, REPRESENTATION_TYPES, MODEL_CONFIGS
from probe_trainer import extract_layer_features
from mlp_classifier import MLPClassifier

def select_salient_neurons(probe, threshold=0.3):
    weights = probe.get_feature_importance()
    total_importance = np.sum(weights)

    sorted_indices = np.argsort(weights)[::-1]
    selected_indices = []
    cumulative_importance = 0.0

    for idx in sorted_indices:
        selected_indices.append(idx)
        cumulative_importance += weights[idx]
        if cumulative_importance >= threshold * total_importance:
            break

    return selected_indices, weights[selected_indices]

def get_layer_weights(probe_results, pooling_type, model_name, layer_weight_strategy="performance"):
    layer_weights = {}
    layer_scores = {}

    num_layers = MODEL_CONFIGS[model_name]["num_layers"]
    for layer_idx in range(num_layers):
        key = f"layer{layer_idx}_{pooling_type}"
        if key in probe_results:
            l_score = probe_results[key]["val_f1"]
            layer_scores[layer_idx] = l_score

    if layer_weight_strategy == "performance":
        max_score = max(layer_scores.values()) if layer_scores else 1.0
        min_score = min(layer_scores.values()) if layer_scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        for layer_idx, score in layer_scores.items():
            normalized_score = (score - min_score) / score_range
            layer_weights[layer_idx] = max(0.1, normalized_score)
    elif layer_weight_strategy == "uniform":
        for layer_idx in layer_scores.keys():
            layer_weights[layer_idx] = 1.0

    return layer_weights

def aggregate_layer_features_weighted(representations, pooling_type, selected_neurons_dict, layer_weights, min_layers=5):
    aggregated_features = []

    sorted_layers = sorted(layer_weights.items(), key=lambda x: x[1], reverse=True)
    selected_layers = sorted_layers[:min_layers] if len(sorted_layers) > min_layers else sorted_layers
    selected_layer_indices = [layer_idx for layer_idx, _ in selected_layers]

    for sample_rep in representations:
        sample_features = []
        for layer_idx in selected_layer_indices:
            layer_features = sample_rep[layer_idx][pooling_type]

            probe_key = f"layer{layer_idx}_{pooling_type}"
            if probe_key in selected_neurons_dict:
                selected_indices = selected_neurons_dict[probe_key]
                selected_features = layer_features[selected_indices]

                weight = layer_weights[layer_idx]
                weighted_features = selected_features * weight
                sample_features.append(weighted_features)

        if sample_features:
            aggregated_features.append(np.concatenate(sample_features))

    return np.array(aggregated_features), selected_layer_indices

def train_aggregated_mlp(train_reps, train_labels, val_reps, val_labels, test_reps, test_labels,
                        best_probes, pooling_type, threshold, min_layers, layer_weight_strategy,
                        hidden_dims, model_name, device="cuda"):

    layer_weights = get_layer_weights(best_probes, pooling_type, model_name, layer_weight_strategy)

    selected_neurons_dict = {}
    for layer_idx in layer_weights.keys():
        key = f"layer{layer_idx}_{pooling_type}"
        if key in best_probes:
            probe = best_probes[key]["probe"]
            selected_indices, _ = select_salient_neurons(probe, threshold)
            selected_neurons_dict[key] = selected_indices

    if not selected_neurons_dict:
        return None

    X_train, selected_layers = aggregate_layer_features_weighted(
        train_reps, pooling_type, selected_neurons_dict, layer_weights, min_layers
    )
    X_val, _ = aggregate_layer_features_weighted(
        val_reps, pooling_type, selected_neurons_dict, layer_weights, min_layers
    )
    X_test, _ = aggregate_layer_features_weighted(
        test_reps, pooling_type, selected_neurons_dict, layer_weights, min_layers
    )

    if len(X_train) == 0:
        return None

    input_dim = X_train.shape[1]

    best_hidden_dim = None
    best_val_f1 = 0
    best_mlp = None

    for hidden_dim in hidden_dims:
        mlp = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        mlp.train(X_train, train_labels, X_val, val_labels, epochs=100, batch_size=64, lr=0.001, patience=15)

        val_f1 = mlp.evaluate(X_val, val_labels, metric='f1_macro')
        print(f"  Hidden dim {hidden_dim}: val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_hidden_dim = hidden_dim
            best_mlp = mlp

    train_f1 = best_mlp.evaluate(X_train, train_labels, metric='f1_macro')
    val_f1 = best_mlp.evaluate(X_val, val_labels, metric='f1_macro')
    test_f1 = best_mlp.evaluate(X_test, test_labels, metric='f1_macro')
    test_acc = best_mlp.evaluate(X_test, test_labels, metric='accuracy')

    return {
        "pooling_type": pooling_type,
        "threshold": threshold,
        "min_layers": min_layers,
        "layer_weight_strategy": layer_weight_strategy,
        "hidden_dim": best_hidden_dim,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "num_features": input_dim,
        "selected_layers": selected_layers,
        "layer_weights": {str(k): v for k, v in layer_weights.items()},
        "final_mlp": best_mlp
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="toxic_chat")
    parser.add_argument("--pooling_types", type=str, nargs="+", default=["residual_mean", "residual_last", "mlp_mean", "mlp_last"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--min_layers", type=int, default=28)
    parser.add_argument("--layer_weight_strategy", type=str, default="performance", choices=["performance", "uniform"])
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Pooling types: {args.pooling_types}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Min layers: {args.min_layers}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Device: {device}")

    with open(f"representations/{args.model}_{args.dataset}_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(f"representations/{args.model}_{args.dataset}_validation.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(f"representations/{args.model}_{args.dataset}_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(f"probes/{args.model}_{args.dataset}_probes.pkl", "rb") as f:
        probe_data = pickle.load(f)

    train_reps = train_data["representations"]
    train_labels = train_data["labels"]
    val_reps = val_data["representations"]
    val_labels = val_data["labels"]
    test_reps = test_data["representations"]
    test_labels = test_data["labels"]
    num_layers = train_data["num_layers"]
    best_probes = probe_data["best_probes"]

    # val_split_ratio = 0.2
    # val_size = int(len(train_labels) * val_split_ratio)
    # val_reps = train_reps[:val_size]
    # val_labels = train_labels[:val_size]
    # train_reps = train_reps[val_size:]
    # train_labels = train_labels[val_size:]

    print(f"\nTrain samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Number of layers: {num_layers}")

    all_results = []
    best_overall_f1 = 0
    best_overall_result = None

    for pooling_type in args.pooling_types:
        for threshold in args.thresholds:
            print(f"\n{'='*60}")
            print(f"Training MLP: pooling={pooling_type}, threshold={threshold}")
            print(f"{'='*60}")

            result = train_aggregated_mlp(
                train_reps, train_labels, val_reps, val_labels, test_reps, test_labels,
                best_probes, pooling_type, threshold, args.min_layers,
                args.layer_weight_strategy, args.hidden_dims, args.model, device
            )

            if result:
                all_results.append(result)
                print(f"Result: val_f1={result['val_f1']:.4f}, test_f1={result['test_f1']:.4f}, test_acc={result['test_acc']:.4f}")

                if result['val_f1'] > best_overall_f1:
                    best_overall_f1 = result['val_f1']
                    best_overall_result = result

    if best_overall_result:
        print(f"\n{'='*60}")
        print(f"BEST SPIN MLP Results:")
        print(f"{'='*60}")
        print(f"Pooling type: {best_overall_result['pooling_type']}")
        print(f"Threshold: {best_overall_result['threshold']}")
        print(f"Hidden dim: {best_overall_result['hidden_dim']}")
        print(f"Features: {best_overall_result['num_features']}")
        print(f"Layers used: {len(best_overall_result['selected_layers'])}")
        print(f"Train F1 (macro): {best_overall_result['train_f1']:.4f}")
        print(f"Val F1 (macro):   {best_overall_result['val_f1']:.4f}")
        print(f"Test F1 (macro):  {best_overall_result['test_f1']:.4f}")
        print(f"Test Accuracy:    {best_overall_result['test_acc']:.4f}")

        output_file = f"probes/{args.model}_{args.dataset}_aggregated_mlp.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(best_overall_result, f)
        print(f"\nSaved best MLP to {output_file}")

        results_file = f"probes/{args.model}_{args.dataset}_aggregated_mlp_results.json"
        json_result = {k: v for k, v in best_overall_result.items() if k != "final_mlp"}
        all_results_json = [
            {k: v for k, v in r.items() if k != "final_mlp"}
            for r in all_results
        ]
        with open(results_file, "w") as f:
            json.dump({"best": json_result, "all_results": all_results_json}, f, indent=2)
        print(f"Saved all results to {results_file}")
    else:
        print("No valid results!")
