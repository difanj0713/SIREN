import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from streaming_extractor import StreamingRepresentationExtractor
from utils.config import MODEL_CONFIGS

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

def get_predictions_batch(representations, siren_model):
    pooling_type = siren_model["pooling_type"]
    selected_neurons_dict = siren_model["selected_neurons_dict"]
    layer_weights = siren_model["layer_weights"]
    selected_layers = siren_model["selected_layers"]

    X = aggregate_features(representations, pooling_type, selected_neurons_dict, layer_weights, selected_layers)

    model = siren_model["final_mlp"]
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        batch_X = torch.FloatTensor(X).to(device)
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    return preds

def test_recall_at_multiple_positions(sample, extractor, tokenizer, siren_model):
    input_ids_full = sample['input_ids']
    unsafe_start = sample['unsafe_start_index']
    unsafe_end = sample['unsafe_end_index']

    messages = sample['message']
    user_text = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=False)
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    assistant_start_token = len(user_ids)

    total_tokens = len(input_ids_full)

    positions = {
        'timely': unsafe_end,
        '1-32': min(total_tokens, unsafe_end + 32),
        '33-64': min(total_tokens, unsafe_end + 64),
        '65-128': min(total_tokens, unsafe_end + 128),
        '129-256': min(total_tokens, unsafe_end + 256),
    }

    results = {}

    for pos_name, pos_idx in positions.items():
        if pos_idx <= assistant_start_token:
            results[pos_name] = False
            continue

        assistant_ids = input_ids_full[assistant_start_token:pos_idx]

        if len(assistant_ids) == 0:
            results[pos_name] = False
            continue

        ids = assistant_ids
        mask = [1] * len(ids)

        input_ids_list = [ids]
        attention_mask_list = [mask]

        reps = extractor.extract_prefixes_batch(input_ids_list, attention_mask_list)
        preds = get_predictions_batch(reps, siren_model)

        results[pos_name] = (preds[0] == 1)

    return results

def evaluate_split(model_name, split_name, device):
    print(f"\n{'='*80}")
    print(f"Streaming Test: {split_name}")
    print(f"{'='*80}")

    dataset = load_dataset("Qwen/Qwen3GuardTest", split=split_name)
    siren_model = load_general_siren(model_name)

    model_config = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extractor = StreamingRepresentationExtractor(
        model_config["model_path"],
        device=device,
        batch_size=32,
        rep_types=["residual_mean", "mlp_mean"]
    )
    extractor.register_hooks()

    results = []

    for sample in tqdm(dataset, desc=f"Processing Streaming Detection"):
        position_detections = test_recall_at_multiple_positions(sample, extractor, tokenizer, siren_model)

        results.append({
            'unique_id': sample['unique_id'],
            'unsafe_start_index': sample['unsafe_start_index'],
            'unsafe_end_index': sample['unsafe_end_index'],
            'detected_timely': position_detections['timely'],
            'detected_1_32': position_detections['1-32'],
            'detected_33_64': position_detections['33-64'],
            'detected_65_128': position_detections['65-128'],
            'detected_129_256': position_detections['129-256'],
            'source': sample['source'],
            'unsafe_type': sample['unsafe_type']
        })

        torch.cuda.empty_cache()

    extractor.remove_hooks()
    del extractor
    torch.cuda.empty_cache()

    detected_timely = sum([r['detected_timely'] for r in results])
    detected_grace_32 = sum([r['detected_timely'] or r['detected_1_32'] for r in results])
    detected_grace_64 = sum([r['detected_timely'] or r['detected_1_32'] or r['detected_33_64'] for r in results])
    detected_grace_128 = sum([r['detected_timely'] or r['detected_1_32'] or r['detected_33_64'] or r['detected_65_128'] for r in results])
    detected_grace_256 = sum([r['detected_timely'] or r['detected_1_32'] or r['detected_33_64'] or r['detected_65_128'] or r['detected_129_256'] for r in results])

    metrics = {
        'split': split_name,
        'num_samples': len(results),
        'timely_recall': detected_timely / len(results) if results else 0,
        'grace_period_32_recall': detected_grace_32 / len(results) if results else 0,
        'grace_period_64_recall': detected_grace_64 / len(results) if results else 0,
        'grace_period_128_recall': detected_grace_128 / len(results) if results else 0,
        'grace_period_256_recall': detected_grace_256 / len(results) if results else 0,
        'timely_count': detected_timely,
        'grace_period_32_count': detected_grace_32,
        'grace_period_64_count': detected_grace_64,
        'grace_period_128_count': detected_grace_128,
        'grace_period_256_count': detected_grace_256,
    }

    print(f"\n{'='*80}")
    print(f"STREAMING DETECTION RESULTS")
    print(f"{'='*80}")
    print(f"Timely (unsafe_end):              {metrics['timely_recall']:.1%} ({metrics['timely_count']}/{len(results)})")
    print(f"Grace Period (1-32 tokens):        {metrics['grace_period_32_recall']:.1%} ({metrics['grace_period_32_count']}/{len(results)})")
    print(f"Grace Period (33-64 tokens):        {metrics['grace_period_64_recall']:.1%} ({metrics['grace_period_64_count']}/{len(results)})")
    print(f"Grace Period (65-128 tokens):       {metrics['grace_period_128_recall']:.1%} ({metrics['grace_period_128_count']}/{len(results)})")
    print(f"Grace Period (129-256 tokens):       {metrics['grace_period_256_recall']:.1%} ({metrics['grace_period_256_count']}/{len(results)})")

    return metrics, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--splits', type=str, nargs='+', default=['thinking_loc'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    all_results = {}

    for split in args.splits:
        metrics, results = evaluate_split(args.model, split, args.device)
        all_results[split] = {
            'metrics': metrics,
            'results': results
        }

    output_dir = f"results/{args.model}"
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/streaming_results.json"

    def convert_json(obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_json(item) for item in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump({
            'model': args.model,
            'results': convert_json(all_results)
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
