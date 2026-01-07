"""
ToxicChat: https://huggingface.co/datasets/lmsys/toxic-chat
OpenAIModeration: https://huggingface.co/datasets/walledai/openai-moderation-dataset
Aegis: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0
Aegis 2.0: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
SimpleSafetyTests: https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests           # Only has test set
HarmBench: https://huggingface.co/datasets/walledai/HarmBench                               # Only has harmful examples
WildguardTest: https://huggingface.co/datasets/allenai/wildguardmix
SafeRLHF: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
Beavertails: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
XSTest: https://huggingface.co/datasets/Paul/XSTest       # Only has test set
"""
from datasets import DatasetDict, load_dataset


def preprocess_dataset(dataset: str, val_ratio: float = 0.2) -> DatasetDict:
    if dataset == "toxic_chat":
        return _preprocess_toxic_chat(val_ratio)
    elif dataset == "openai_moderation":
        return _preprocess_openai_moderation(val_ratio)
    elif dataset == "aegis":
        return _preprocess_aegis(val_ratio)
    elif dataset == "aegis2":
        return _preprocess_aegis2(val_ratio)
    elif dataset == "wildguard":
        return _preprocess_wildguard(val_ratio)
    elif dataset == "safe_rlhf":
        return _preprocess_safe_rlhf(val_ratio)
    elif dataset == "beavertails":
        return _preprocess_beavertails(val_ratio)
    elif dataset == "xstest":
        return _preprocess_xstest(val_ratio)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def _preprocess_toxic_chat(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

    split = ds["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )

    for k in out.keys():
        out[k] = out[k].rename_column("user_input", "text")
        out[k] = out[k].rename_column("toxicity", "label")

    return out


def _preprocess_openai_moderation(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("walledai/openai-moderation-dataset")

    test_val_size = val_ratio * 2
    split_60_40 = ds["train"].train_test_split(test_size=test_val_size, shuffle=True)
    split_20_20 = split_60_40["test"].train_test_split(test_size=0.5, shuffle=True)
    out = DatasetDict(
        {
            "train": split_60_40["train"],
            "validation": split_20_20["train"],
            "test": split_20_20["test"],
        }
    )

    label_cols = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]

    def to_binary_label(batch):
        n = len(batch[label_cols[0]])
        labels = []
        for i in range(n):
            val = 0
            for c in label_cols:
                if batch[c][i] == 1:
                    val = 1
                    break
            labels.append(val)
        return {"label": labels}

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].map(to_binary_label, batched=True)

    return out


def _preprocess_aegis(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")

    split = ds["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )

    label_cols = ["labels_0", "labels_1", "labels_2", "labels_3", "labels_4"]

    def convert_and_vote(batch):
        n = len(batch[label_cols[0]])
        cols = {c: [] for c in label_cols}
        final_labels = []

        for i in range(n):
            safe_cnt = 0
            unsafe_cnt = 0

            for c in label_cols:
                v = batch[c][i]
                if v is None:
                    cols[c].append(None)
                    continue
                if v == "Safe":
                    cols[c].append(0)
                    safe_cnt += 1
                else:
                    cols[c].append(1)
                    unsafe_cnt += 1

            if safe_cnt == 0 and unsafe_cnt == 0:
                final_labels.append(None)
            else:
                # tie -> 1
                final_labels.append(1 if unsafe_cnt >= safe_cnt else 0)

        cols["label"] = final_labels
        return cols

    for k in out.keys():
        out[k] = out[k].map(convert_and_vote, batched=True)
        none_count = out[k].filter(lambda x: x["label"] is None).num_rows
        assert none_count == 0

    return out


def _preprocess_aegis2(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
    out = DatasetDict(
        {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["test"],
        }
    )

    def to_binary_label(example):
        v = example["label"]
        if v == "unsafe":
            return {"label": 1}
        if v == "safe":
            return {"label": 0}
        raise ValueError(f"Invalid dataset: {v}")

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].rename_column("prompt_label", "label")
        out[k] = out[k].map(to_binary_label)

    return out


def _preprocess_wildguard(val_ratio: float = 0.2) -> DatasetDict:
    train_ds = load_dataset("allenai/wildguardmix", "wildguardtrain")
    test_ds = load_dataset("allenai/wildguardmix", "wildguardtest")

    train_split = train_ds["train"]
    test_split = test_ds["test"]

    split = train_split.train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict({"train": split["train"], "validation": split["test"], "test": test_split})

    def to_binary_label(example):
        v = example["label"]
        if v is None:
            return {"label": None}
        if v == "unharmful":
            return {"label": 0}
        if v == "harmful":
            return {"label": 1}
        raise ValueError(f"Invalid dataset: {v}")

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].rename_column("prompt_harm_label", "label")
        out[k] = out[k].map(to_binary_label)
        out[k] = out[k].filter(lambda x: x["label"] is not None)

    return out


def _preprocess_safe_rlhf(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")

    split = ds["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )

    def to_binary_label(example):
        v = example["label"]
        if v is True:
            label = 0
        elif v is False:
            label = 1
        else:
            raise ValueError(f"Invalid dataset: {v}")

        return {"label": label, "text": f"{example['prompt']}\n{example['response_0']}"}

    for k in out.keys():
        out[k] = out[k].rename_column("is_response_0_safe", "label")
        out[k] = out[k].map(to_binary_label)

    return out


def _preprocess_beavertails(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("PKU-Alignment/BeaverTails")
    base = DatasetDict({"train": ds["30k_train"], "test": ds["30k_test"]})

    split = base["train"].train_test_split(test_size=val_ratio, shuffle=True)
    out = DatasetDict({"train": split["train"], "validation": split["test"], "test": base["test"]})

    def to_label_and_text(example):
        v = example["label"]
        if v is True:
            label = 0
        elif v is False:
            label = 1
        else:
            raise ValueError(f"Invalid dataset: {v}")
        return {"label": label, "text": f"{example['prompt']}\n{example['response']}"}

    for k in out.keys():
        out[k] = out[k].rename_column("is_safe", "label")
        out[k] = out[k].map(to_label_and_text)

    return out


def _preprocess_xstest(val_ratio: float = 0.2) -> DatasetDict:
    ds = load_dataset("Paul/XSTest")

    test_val_size = val_ratio * 2
    split_60_40 = ds["train"].train_test_split(test_size=test_val_size, shuffle=True)
    split_20_20 = split_60_40["test"].train_test_split(test_size=0.5, shuffle=True)
    out = DatasetDict(
        {
            "train": split_60_40["train"],
            "validation": split_20_20["train"],
            "test": split_20_20["test"],
        }
    )

    def to_binary_label(example):
        v = example["label"]
        if v == "safe":
            return {"label": 0}
        if v == "unsafe":
            return {"label": 1}
        raise ValueError(f"Invalid dataset: {v}")

    for k in out.keys():
        out[k] = out[k].rename_column("prompt", "text")
        out[k] = out[k].map(to_binary_label)

    return out
