import sys
sys.path.append('../spin')

import argparse
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from config import DATASET_CONFIGS, MODEL_CONFIGS
from model_hooks import Qwen3RepresentationExtractor
from preprocess import preprocess_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-0.6b")
parser.add_argument("--dataset", type=str, default="toxic_chat")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--rep_types", type=str, nargs="+", default=["residual_mean", "mlp_mean"])
args = parser.parse_args()

model_config = MODEL_CONFIGS[args.model]

print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
print(f"Batch size: {args.batch_size}")
print(f"Number of layers: {model_config['num_layers']}")

extractor = Qwen3RepresentationExtractor(
    model_config["model_path"],
    device=args.device,
    batch_size=args.batch_size,
    rep_types=args.rep_types
)
extractor.register_hooks()
dataset = preprocess_dataset(args.dataset)

for split_name, split_data in dataset.items():
    print(f"\nProcessing {split_name} split...")

    all_representations = []
    all_labels = []

    batch_texts = []
    batch_labels = []

    for idx, item in enumerate(tqdm(split_data)):
        text = item["text"]
        label = item["label"]

        batch_texts.append(text)
        batch_labels.append(label)

        if len(batch_texts) == args.batch_size or idx == len(split_data) - 1:
            batch_reps = extractor.extract_batch(batch_texts)
            all_representations.extend(batch_reps)
            all_labels.extend(batch_labels)

            batch_texts = []
            batch_labels = []

    output = {
        "representations": all_representations,
        "labels": np.array(all_labels),
        "num_layers": model_config["num_layers"],
        "num_samples": len(all_labels),
        "dataset": args.dataset,
        "model": args.model,
        "split": split_name
    }

    output_file = f"representations/{args.model}_{args.dataset}_{split_name}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved {len(all_labels)} samples to {output_file}")

extractor.remove_hooks()
print("\nDone!")
