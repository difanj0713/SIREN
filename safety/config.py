DATASET_CONFIGS = {
    "toxic_chat": {
        "hf_name": "lmsys/toxic-chat",
        "hf_config": "toxicchat0124",
        "splits": {
            "train": "train",
            "test": "test"
        },
        "text_field": "user_input",
        "label_field": "toxicity",
        "num_classes": 2,
        "label_names": ["benign", "toxic"]
    }
}

MODEL_CONFIGS = {
    "qwen3-0.6b": {
        "model_path": "Qwen/Qwen3-0.6B",
        "num_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072
    }
}

POOLING_STRATEGIES = ["mean", "last"]
REPRESENTATION_TYPES = ["residual", "mlp"]
