from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import argparse

from dataset_helper import get_dataset_config
from model_helper import get_model_config, format_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--judge_model", type=str, default="qwen2.5-7b")
parser.add_argument("--gen_model", type=str, default="qwen3-0.6b")
parser.add_argument("--dataset", type=str, default="tweet_hate")
parser.add_argument("--tensor_parallel_size", type=int, default=2)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
args = parser.parse_args()

judge_model_config = get_model_config(args.judge_model)
dataset_config = get_dataset_config(args.dataset)

llm = LLM(
    model=judge_model_config["model_path"],
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True
)

# input_file = f"data/{args.gen_model}_{args.dataset}_spin_generations.jsonl"
input_file = f"data/{args.gen_model}_{args.dataset}_generations.jsonl"
with open(input_file, "r") as f:
    generations = [json.loads(line) for line in f]

judge_prompts = []
for gen in generations:
    user_prompt = dataset_config["judge_prompt"].format(response=gen["generated_response"])
    prompt = format_prompt(llm.get_tokenizer(), judge_model_config, user_prompt)
    judge_prompts.append(prompt)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
    top_p=1.0
)

print(f"Judge model: {args.judge_model}")
print(f"Dataset: {args.dataset}")
print(f"Judging {len(judge_prompts)} responses...")
outputs = llm.generate(judge_prompts, sampling_params)

final_results = []
correct = 0
total = 0
valid_labels = set(dataset_config["judge_labels"])

for idx, output in enumerate(tqdm(outputs)):
    predicted_label = output.outputs[0].text.strip().lower()

    for label in valid_labels:
        if label.replace("_", " ") in predicted_label or label in predicted_label:
            predicted_label = label
            break

    true_label = generations[idx]["true_label"]
    is_correct = predicted_label == true_label

    if is_correct:
        correct += 1
    total += 1

    final_results.append({
        "id": generations[idx]["id"],
        "text": generations[idx]["text"],
        "true_label": true_label,
        "generated_response": generations[idx]["generated_response"],
        "predicted_label": predicted_label,
        "correct": is_correct
    })

accuracy = correct / total
print(f"\n{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"{'='*60}")

output_file = f"data/{args.gen_model}_{args.dataset}_final_results.jsonl"
summary_file = f"data/{args.gen_model}_{args.dataset}_summary.txt"

with open(output_file, "w") as f:
    for result in final_results:
        f.write(json.dumps(result) + "\n")

with open(summary_file, "w") as f:
    f.write(f"Model: {args.gen_model}\n")
    f.write(f"Dataset: {args.dataset}\n")
    f.write(f"Judge: {args.judge_model}\n")
    f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Total: {total}\n")

print(f"\nResults saved to {output_file}")
print(f"Summary saved to {summary_file}")
