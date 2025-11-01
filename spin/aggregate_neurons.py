import argparse
import pickle
import json
import numpy as np
import torch

from config import POOLING_STRATEGIES, REPRESENTATION_TYPES, MODEL_CONFIGS
from probe_trainer import extract_layer_features, LinearProbe

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
    layer_accs = {}
    
    num_layers = MODEL_CONFIGS[model_name]["num_layers"]
    for layer_idx in range(num_layers):
        key = f"layer{layer_idx}_{pooling_type}"
        if key in probe_results:
            l_acc = probe_results[key]["test_acc"]
            layer_accs[layer_idx] = l_acc
    
    if layer_weight_strategy == "performance":
        max_acc = max(layer_accs.values()) if layer_accs else 1.0
        min_acc = min(layer_accs.values()) if layer_accs else 0.0
        acc_range = max_acc - min_acc if max_acc > min_acc else 1.0
        
        for layer_idx, acc in layer_accs.items():
            normalized_acc = (acc - min_acc) / acc_range
            layer_weights[layer_idx] = max(0.1, normalized_acc)
    elif layer_weight_strategy == "uniform":
        for layer_idx in layer_accs.keys():
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

def train_aggregated_classifier(train_reps, train_labels, val_reps, val_labels, test_reps, test_labels, 
                               best_probes, pooling_type, threshold, min_layers, layer_weight_strategy, 
                               c_values, model_name, device="cuda"):
    
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
    
    best_C = None
    best_val_acc = 0
    
    for C in c_values:
        probe = LinearProbe(C=C, device=device)
        probe.train(X_train, train_labels, X_val, val_labels, quick_eval=True)
        val_acc = probe.evaluate(X_val, val_labels)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
    
    final_probe = LinearProbe(C=best_C, device=device)
    final_probe.train(X_train, train_labels, X_val, val_labels, quick_eval=False)
    
    train_acc = final_probe.evaluate(X_train, train_labels)
    val_acc = final_probe.evaluate(X_val, val_labels)
    test_acc = final_probe.evaluate(X_test, test_labels)
    
    return {
        "pooling_type": pooling_type,
        "threshold": threshold,
        "min_layers": min_layers,
        "layer_weight_strategy": layer_weight_strategy,
        "best_C": best_C,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "num_features": X_train.shape[1],
        "selected_layers": selected_layers,
        "layer_weights": {str(k): v for k, v in layer_weights.items()},
        "final_probe": final_probe
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="tweet_hate")
    parser.add_argument("--pooling_type", type=str, default="residual_mean")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min_layers", type=int, default=28)
    parser.add_argument("--layer_weight_strategy", type=str, default="performance", choices=["performance", "uniform"])
    parser.add_argument("--c_values", type=float, nargs="+", default=[1.0, 5.0, 10.0, 50, 100.0])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Pooling type: {args.pooling_type}")
    print(f"Threshold: {args.threshold}")
    print(f"Min layers: {args.min_layers}")
    print(f"Layer weight strategy: {args.layer_weight_strategy}")
    print(f"C values: {args.c_values}")
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

    print(f"\nTrain samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Number of layers: {num_layers}")

    print(f"\nTraining aggregated SPIN classifier...")
    result = train_aggregated_classifier(
        train_reps, train_labels, val_reps, val_labels, test_reps, test_labels,
        best_probes, args.pooling_type, args.threshold, args.min_layers, 
        args.layer_weight_strategy, args.c_values, args.model, device
    )

    if result:
        print(f"\n{'='*60}")
        print(f"SPIN Aggregated Results:")
        print(f"{'='*60}")
        print(f"Pooling type: {result['pooling_type']}")
        print(f"Selected layers: {result['selected_layers']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Layer weight strategy: {result['layer_weight_strategy']}")
        print(f"Best C: {result['best_C']}")
        print(f"Features: {result['num_features']}")
        print(f"Train accuracy: {result['train_acc']:.4f}")
        print(f"Validation accuracy: {result['val_acc']:.4f}")
        print(f"Test accuracy: {result['test_acc']:.4f}")
        
        print(f"\nLayer weights:")
        for layer_str, weight in result['layer_weights'].items():
            layer_idx = int(layer_str)
            if layer_idx in result['selected_layers']:
                print(f"  Layer {layer_idx}: {weight:.3f} ✓")
        
        output_file = f"probes/{args.model}_{args.dataset}_aggregated.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(result, f)
        print(f"\nSaved aggregated results to {output_file}")
        
        results_file = f"probes/{args.model}_{args.dataset}_aggregated_results.json"
        json_result = {k: v for k, v in result.items() if k != "final_probe"}
        with open(results_file, "w") as f:
            json.dump(json_result, f, indent=2)
        print(f"Saved results summary to {results_file}")
    else:
        print("Failed to train aggregated classifier!")