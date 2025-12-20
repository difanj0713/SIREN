import sys
sys.path.append('..')
import os
import json
import pickle
import argparse
from datasets import load_dataset
from config import DETOXIFY_PROMPT, REWRITER_CONFIGS

def load_toxic_samples(dataset_name):
    if dataset_name == "toxic_chat":
        dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
        test_data = dataset["test"]
        toxic_samples = []
        for idx in range(len(test_data)):
            if test_data[idx]["toxicity"] == 1:
                toxic_samples.append({
                    "id": idx,
                    "original": test_data[idx]["user_input"],
                    "label": 1
                })
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return toxic_samples

def generate_local(samples, config):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=config["tensor_parallel_size"],
        gpu_memory_utilization=0.9,
        max_model_len=32768
    )

    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"]
    )

    prompts = []
    for sample in samples:
        prompt = DETOXIFY_PROMPT.format(text=sample["original"])
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        results.append({
            "id": samples[i]["id"],
            "original": samples[i]["original"],
            "detoxified": output.outputs[0].text.strip(),
            "label": samples[i]["label"]
        })

    return results

def generate_openrouter(samples, config):
    import requests
    from tqdm import tqdm

    results = []
    for sample in tqdm(samples, desc="Paraphrasing"):
        prompt = DETOXIFY_PROMPT.format(text=sample["original"])

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config["model_name"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"]
                },
                timeout=60
            )

            response.raise_for_status()
            response_data = response.json()
            detoxified = response_data["choices"][0]["message"]["content"].strip()

            results.append({
                "id": sample["id"],
                "original": sample["original"],
                "detoxified": detoxified,
                "label": sample["label"]
            })
        except Exception as e:
            print(f"\nError on sample {sample['id']}: {e}")
            results.append({
                "id": sample["id"],
                "original": sample["original"],
                "detoxified": f"[ERROR: {str(e)}]",
                "label": sample["label"]
            })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--rewriter", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading toxic samples from {args.dataset}...")
    toxic_samples = load_toxic_samples(args.dataset)
    print(f"Loaded {len(toxic_samples)} toxic samples")

    config = REWRITER_CONFIGS[args.rewriter]

    print(f"Generating detoxified versions using {args.rewriter} ({config['type']})...")
    if config["type"] == "local":
        results = generate_local(toxic_samples, config)
    elif config["type"] == "openrouter":
        results = generate_openrouter(toxic_samples, config)
    else:
        raise ValueError(f"Unknown rewriter type: {config['type']}")

    os.makedirs(f"rewrite/{args.dataset}", exist_ok=True)
    output_path = f"rewrite/{args.dataset}/{args.rewriter}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {output_path}")

    for sample in results[:3]:
        print(f"\nID: {sample['id']}")
        print(f"Original:   {sample['original']}")
        print(f"Detoxified: {sample['detoxified']}")

if __name__ == "__main__":
    main()
