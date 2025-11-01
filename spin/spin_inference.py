import argparse
import pickle
import json
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

from config import DATASET_CONFIGS, MODEL_CONFIGS
from model_hooks import Qwen3RepresentationExtractor
from aggregate_neurons import aggregate_layer_features_weighted, select_salient_neurons, get_layer_weights

def load_aggregated_classifier(model_name, dataset_name):
    with open(f"probes/{model_name}_{dataset_name}_aggregated.pkl", "rb") as f:
        agg_data = pickle.load(f)
    
    with open(f"probes/{model_name}_{dataset_name}_probes.pkl", "rb") as f:
        probe_data = pickle.load(f)
    
    best_probes = probe_data["best_probes"]
    final_probe = agg_data["final_probe"]
    pooling_type = agg_data["pooling_type"]
    threshold = agg_data["threshold"]
    min_layers = agg_data["min_layers"]
    layer_weight_strategy = agg_data["layer_weight_strategy"]
    selected_layers = agg_data["selected_layers"]
    
    layer_weights = get_layer_weights(best_probes, pooling_type, model_name, layer_weight_strategy)
    
    selected_neurons_dict = {}
    for layer_idx in selected_layers:
        key = f"layer{layer_idx}_{pooling_type}"
        if key in best_probes:
            probe = best_probes[key]["probe"]
            selected_indices, _ = select_salient_neurons(probe, threshold)
            selected_neurons_dict[key] = selected_indices
    
    return final_probe, pooling_type, selected_neurons_dict, layer_weights, min_layers, agg_data

def generate_verbalization(spin_logit, label_names):
    prob = spin_logit
    if prob > 0.5:
        predicted_class = label_names[1]
        confidence = int(round(prob * 100))
    else:
        predicted_class = label_names[0]
        confidence = int(round((1 - prob) * 100))

    if label_names == ["not_hate", "hate"]:
        if predicted_class == "hate":
            return f"Based on your internal reasoning, this tweet has approximately {confidence}% likelihood of containing hate speech."
        else:
            return f"Based on your internal reasoning, this tweet has approximately {confidence}% likelihood of NOT containing hate speech."
    else:
        return f"Based on your internal reasoning, this text has approximately {confidence}% probability of being {predicted_class.upper()}."

def run_spin_inference(model_name="qwen3-0.6b", dataset_name="tweet_hate", device="cuda", batch_size=128):
    print(f"Loading SPIN classifier for {model_name} on {dataset_name}...")
    final_probe, pooling_type, selected_neurons_dict, layer_weights, min_layers, agg_data = load_aggregated_classifier(model_name, dataset_name)
    
    print(f"SPIN config: pooling={pooling_type}, threshold={agg_data['threshold']:.2f}, "
          f"layers={len(agg_data['selected_layers'])}, test_acc={agg_data['test_acc']:.4f}")
    
    model_config = MODEL_CONFIGS[model_name]
    dataset_config = DATASET_CONFIGS[dataset_name]

    print(f"\nExtracting representations from {model_config['model_path']}...")
    extractor = Qwen3RepresentationExtractor(model_config["model_path"], device=device, batch_size=batch_size)
    extractor.register_hooks()

    if dataset_config["hf_config"]:
        dataset = load_dataset(dataset_config["hf_name"], dataset_config["hf_config"])
    else:
        dataset = load_dataset(dataset_config["hf_name"])

    test_data = dataset[dataset_config["splits"]["test"]]

    results = []
    batch_texts = []
    batch_items = []

    print(f"\nRunning SPIN inference on {len(test_data)} test samples...")

    for idx, item in enumerate(tqdm(test_data)):
        text = item[dataset_config["text_field"]]
        label = item[dataset_config["label_field"]]

        batch_texts.append(text)
        batch_items.append({"idx": idx, "text": text, "label": label})

        if len(batch_texts) == batch_size or idx == len(test_data) - 1:
            batch_reps = extractor.extract_batch(batch_texts)

            for rep, item_data in zip(batch_reps, batch_items):
                feat, _ = aggregate_layer_features_weighted(
                    [rep], pooling_type, selected_neurons_dict, layer_weights, min_layers
                )
                
                if len(feat) > 0:
                    final_probe.model.eval()
                    with torch.no_grad():
                        feat_tensor = torch.FloatTensor(feat).to(device)
                        outputs = final_probe.model(feat_tensor)
                        # For binary classification, logit = difference between class logits
                        raw_logit = (outputs[0][1] - outputs[0][0]).cpu().item()
                        # Convert to probability using sigmoid
                        prob = torch.sigmoid(torch.tensor(raw_logit)).item()
                    
                    verbalization = generate_verbalization(prob, dataset_config["label_names"])

                    results.append({
                        "id": item_data["idx"],
                        "text": item_data["text"],
                        "label": dataset_config["label_names"][item_data["label"]],
                        "spin_logit": float(prob),
                        "spin_verbalization": verbalization
                    })

            batch_texts = []
            batch_items = []

    extractor.remove_hooks()

    output_file = f"../llm-inference/data/{model_name}_{dataset_name}_spin_augmented.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} SPIN-augmented samples to {output_file}")
    print(f"\nSample verbalization:")
    if results:
        print(f"Text: {results[0]['text'][:100]}...")
        print(f"SPIN: {results[0]['spin_verbalization']}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="tweet_hate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_spin_inference(args.model, args.dataset, args.device, args.batch_size)