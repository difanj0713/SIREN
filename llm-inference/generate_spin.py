from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import argparse

from dataset_helper import get_dataset_config
from model_helper import get_model_config, format_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-0.6b")
parser.add_argument("--dataset", type=str, default="tweet_hate")
parser.add_argument("--tensor_parallel_size", type=int, default=2)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
parser.add_argument("--max_tokens", type=int, default=512)
args = parser.parse_args()

model_config = get_model_config(args.model)
dataset_config = get_dataset_config(args.dataset)

spin_file = f"data/{args.model}_{args.dataset}_spin_augmented.json"
with open(spin_file, "r") as f:
    spin_data = json.load(f)

spin_dict = {item["id"]: item for item in spin_data}

llm = LLM(
    model=model_config["model_path"],
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True
)

prompts = []
samples = []
for item in spin_data:
    base_prompt = dataset_config["prompt_template"].format(text=item["text"], spin_verbalization=item['spin_verbalization'])
    # augmented_prompt = f"{base_prompt}\n\n{item['spin_verbalization']}"
    prompt = format_prompt(llm.get_tokenizer(), model_config, base_prompt)
    prompts.append(prompt)
    samples.append(item)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=args.max_tokens,
    top_p=1.0,
    stop=dataset_config["stop_words"],
    include_stop_str_in_output=True
)

print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
print(f"Generating SPIN-augmented responses for {len(prompts)} samples...")
outputs = llm.generate(prompts, sampling_params)

results = []
for idx, output in enumerate(tqdm(outputs)):
    generated_text = output.outputs[0].text
    results.append({
        "id": samples[idx]["id"],
        "text": samples[idx]["text"],
        "true_label": samples[idx]["label"],
        "spin_logit": samples[idx]["spin_logit"],
        "spin_verbalization": samples[idx]["spin_verbalization"],
        "generated_response": generated_text
    })

output_file = f"data/{args.model}_{args.dataset}_spin_generations.jsonl"
with open(output_file, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Saved {len(results)} SPIN-augmented generations to {output_file}")
