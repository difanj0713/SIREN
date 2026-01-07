import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from preprocess_think import preprocess_think_by_backbone
from utils.config import MODEL_CONFIGS
from utils.model_hooks import Qwen3RepresentationExtractor

class AdaptiveMLPClassifier(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout_rates, num_classes=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, (hidden_dim, dropout) in enumerate(zip(layer_dims, dropout_rates)):
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        final_linear = nn.Linear(prev_dim, num_classes)
        nn.init.kaiming_normal_(final_linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def load_general_siren(model_name):
    model_path = f"../../train/probes/optuna/{model_name}_general/best_model.pkl"
    with open(model_path, 'rb') as f:
        siren_model = pickle.load(f)
    return siren_model

def extract_representations(texts, model_name, device, batch_size=256):
    model_config = MODEL_CONFIGS[model_name]
    extractor = Qwen3RepresentationExtractor(
        model_config["model_path"],
        device=device,
        batch_size=batch_size,
        rep_types=["residual_mean", "mlp_mean"]
    )
    extractor.register_hooks()

    representations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting", leave=False):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            batch_reps = extractor.extract_batch(batch_texts)
            representations.extend(batch_reps)
            torch.cuda.empty_cache()

    extractor.remove_hooks()
    del extractor
    torch.cuda.empty_cache()

    return representations

def aggregate_features(representations, pooling_type, selected_neurons_dict, layer_weights, selected_layers):
    aggregated_features = []
    for sample_rep in representations:
        sample_features = []
        for layer_idx in selected_layers:
            key = f"layer{layer_idx}_{pooling_type}"
            if key not in selected_neurons_dict:
                continue
            layer_features = sample_rep[layer_idx][pooling_type]
            selected_indices = selected_neurons_dict[key]
            selected_features = layer_features[selected_indices]
            weight = layer_weights[str(layer_idx)]
            weighted_features = selected_features * weight
            sample_features.append(weighted_features)
        if sample_features:
            aggregated_features.append(np.concatenate(sample_features))
    return np.array(aggregated_features)

def get_siren_predictions(representations, siren_model, pooling_type, selected_neurons_dict):
    layer_weights = siren_model["layer_weights"]
    selected_layers = siren_model["selected_layers"]
    X = aggregate_features(representations, pooling_type, selected_neurons_dict, layer_weights, selected_layers)

    predictions = []
    batch_size = 256
    model = siren_model["final_mlp"]
    model.eval()

    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
            outputs = model(batch_X)
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(batch_preds)
    return np.array(predictions)

def evaluate_backbone(backbone_name, test_data, model_name, siren_model, device, batch_size):
    pooling_type = siren_model["pooling_type"]
    selected_neurons_dict = siren_model["selected_neurons_dict"]

    texts = [sample["text"] for sample in test_data]
    labels = np.array([sample["label"] for sample in test_data])

    representations = extract_representations(texts, model_name, device, batch_size)
    predictions = get_siren_predictions(representations, siren_model, pooling_type, selected_neurons_dict)

    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(labels, predictions, average='binary', pos_label=1, zero_division=0)

    print(f"  F1 Macro: {f1_macro:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return {
        "backbone": backbone_name,
        "num_samples": len(texts),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print(f"Loading General SIREN for {args.model}...")
    siren_model = load_general_siren(args.model)
    print(f"Loaded: pooling={siren_model['pooling_type']}, features={siren_model['num_features']}")

    print("\nPreprocessing think models by backbone...")
    backbone_datasets = preprocess_think_by_backbone()

    all_results = []
    for backbone in ["GLM", "Qwen3", "Deepseek"]:
        if backbone not in backbone_datasets:
            continue

        print(f"\n{'='*60}")
        print(f"Backbone: {backbone}")
        print(f"{'='*60}")

        test_data = backbone_datasets[backbone]["test"]
        result = evaluate_backbone(backbone, test_data, args.model, siren_model, args.device, args.batch_size)
        result["model"] = args.model
        all_results.append(result)

    avg_f1_macro = np.mean([r["f1_macro"] for r in all_results])
    avg_accuracy = np.mean([r["accuracy"] for r in all_results])
    avg_precision = np.mean([r["precision"] for r in all_results])
    avg_recall = np.mean([r["recall"] for r in all_results])

    print("="*80)
    for r in all_results:
        print(f"\n{r['backbone']}:")
        print(f"  F1 Macro:  {r['f1_macro']:.4f}")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")

    print(f"\nOverall Average:")
    print(f"  F1 Macro:  {avg_f1_macro:.4f}")
    print(f"  Accuracy:  {avg_accuracy:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")

    output_dir = f"../../train/probes/optuna/{args.model}_general"
    output_path = f"{output_dir}/think_eval_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "model": args.model,
            "results": all_results,
            "averages": {
                "f1_macro": float(avg_f1_macro),
                "accuracy": float(avg_accuracy),
                "precision": float(avg_precision),
                "recall": float(avg_recall)
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
