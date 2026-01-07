from datasets import Dataset, DatasetDict, load_dataset

def preprocess_think_by_backbone():
    ds = load_dataset("Qwen/Qwen3GuardTest", split="thinking")

    backbone_samples = {"GLM": [], "Qwen3": [], "Deepseek": []}

    for sample in ds:
        source = sample["source"]

        if source.startswith("deepseek-ai/"):
            backbone = "Deepseek"
        elif "GLM" in source:
            backbone = "GLM"
        elif "Qwen3" in source:
            backbone = "Qwen3"
        else:
            continue

        messages = sample["message"]
        user_content = ""
        assistant_content = ""

        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        label_val = sample["label"].lower()
        if label_val == "safe":
            label = 0
        elif label_val == "unsafe":
            label = 1
        else:
            continue

        backbone_samples[backbone].append({
            "text": assistant_content,
            "prompt": user_content,
            "label": label,
            "source": source
        })

    datasets = {}
    for backbone, samples in backbone_samples.items():
        if samples:
            datasets[backbone] = DatasetDict({"test": Dataset.from_list(samples)})

    return datasets
