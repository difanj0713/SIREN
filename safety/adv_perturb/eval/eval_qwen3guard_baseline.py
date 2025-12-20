import sys
sys.path.append('../../')
import argparse
import numpy as np
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from preprocess import preprocess_dataset

def extract_qwen3guard_label(content, mode='strict'):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    match = re.search(safe_pattern, content)
    if match:
        label = match.group(1)
        if label == "Unsafe":
            return 1
        elif label == "Controversial":
            return 1 if mode == 'strict' else 0
        else:
            return 0
    return 0

def get_qwen3guard_raw_outputs(texts, model, tokenizer, device, batch_size=16):
    raw_outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i+batch_size]
        batch_messages = [[{"role": "user", "content": text}] for text in batch_texts]
        batch_prompts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in batch_messages]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        for j, output in enumerate(outputs):
            output_ids = output[len(inputs.input_ids[j]):].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True)
            raw_outputs.append(content)
    return raw_outputs

def parse_predictions(raw_outputs, mode='strict'):
    predictions = []
    for content in raw_outputs:
        label = extract_qwen3guard_label(content, mode=mode)
        predictions.append(label)
    return np.array(predictions)

def evaluate_dataset(dataset_name, model_name, model, tokenizer, device, batch_size=16):
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"{'='*80}")

    dataset_dict = preprocess_dataset(dataset_name)
    test_data = dataset_dict["test"]

    texts = [sample["text"] for sample in test_data]
    labels = np.array([sample["label"] for sample in test_data])

    print(f"Test samples: {len(texts)}")
    print(f"Positive samples: {np.sum(labels == 1)}")
    print(f"Negative samples: {np.sum(labels == 0)}")

    raw_outputs = get_qwen3guard_raw_outputs(texts, model, tokenizer, device, batch_size=batch_size)
    pred_strict = parse_predictions(raw_outputs, mode='strict')
    pred_loose = parse_predictions(raw_outputs, mode='loose')

    def compute_metrics(predictions, labels):
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, average='binary', pos_label=1, zero_division=0)),
            "recall": float(recall_score(labels, predictions, average='binary', pos_label=1, zero_division=0)),
            "f1_binary": float(f1_score(labels, predictions, average='binary', pos_label=1, zero_division=0)),
            "f1_macro": float(f1_score(labels, predictions, average='macro', zero_division=0))
        }

    metrics_strict = compute_metrics(pred_strict, labels)
    metrics_loose = compute_metrics(pred_loose, labels)

    avg_f1_macro = (metrics_strict["f1_macro"] + metrics_loose["f1_macro"]) / 2

    print(f"\nStrict Mode (Controversial=Unsafe):")
    print(f"  F1 Binary: {metrics_strict['f1_binary']:.4f}")
    print(f"  F1 Macro:  {metrics_strict['f1_macro']:.4f}")
    print(f"\nLoose Mode (Controversial=Safe):")
    print(f"  F1 Binary: {metrics_loose['f1_binary']:.4f}")
    print(f"  F1 Macro:  {metrics_loose['f1_macro']:.4f}")
    print(f"\nAverage F1 Macro: {avg_f1_macro:.4f}")

    return {
        "dataset": dataset_name,
        "num_samples": len(texts),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
        "strict": metrics_strict,
        "loose": metrics_loose,
        "avg_f1_macro": float(avg_f1_macro)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3guard-0.6b', 'qwen3guard-4b'])
    parser.add_argument('--datasets', type=str, nargs='+', default=['toxic_chat'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if args.model == 'qwen3guard-0.6b':
        model_path = "Qwen/Qwen3Guard-Gen-0.6B"
    else:
        model_path = "Qwen/Qwen3Guard-Gen-4B"

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device}
    )
    model.eval()

    all_results = []
    for dataset_name in args.datasets:
        result = evaluate_dataset(dataset_name, args.model, model, tokenizer, args.device, args.batch_size)
        all_results.append(result)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} | {'Strict':>7} | {'Loose':>7} | {'Avg':>7}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['dataset']:<20} | {result['strict']['f1_macro']:7.4f} | {result['loose']['f1_macro']:7.4f} | {result['avg_f1_macro']:7.4f}")

    import json
    output_path = f"results/{args.model}_baseline.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
