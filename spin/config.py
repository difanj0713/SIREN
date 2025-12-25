DATASET_CONFIGS = {
    "tweet_hate": {
        "hf_name": "tweet_eval",
        "hf_config": "hate",
        "splits": {
            "train": "train",
            "validation": "validation",
            "test": "test"
        },
        "text_field": "text",
        "label_field": "label",
        "num_classes": 2,
        "label_names": ["not_hate", "hate"]
    },
    "imdb": {
        "hf_name": "stanfordnlp/imdb",
        "hf_config": None,
        "splits": {
            "train": "train",
            "test": "test"
        },
        "text_field": "text",
        "label_field": "label",
        "num_classes": 2,
        "label_names": ["negative", "positive"]
    }
}

MODEL_CONFIGS = {
    "qwen3-0.6b": {
        "model_path": "Qwen/Qwen3-0.6B",
        "model_type": "qwen3",
        "num_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072
    },
    "qwen3-1.7b": {
        "model_path": "Qwen/Qwen3-1.7B",
        "model_type": "qwen3",
        "num_layers": 28,
        "hidden_size": 2048,
        "intermediate_size": 5504
    },
    "qwen3-4b": {
        "model_path": "Qwen/Qwen3-4B-Instruct-2507",
        "model_type": "qwen3",
        "num_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728
    },
    "qwen3guard-0.6b": {
        "model_path": "Qwen/Qwen3Guard-Gen-0.6B",
        "model_type": "qwen3",
        "num_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072
    },
    "qwen3guard-4b": {
        "model_path": "Qwen/Qwen3Guard-Gen-4B",
        "model_type": "qwen3",
        "num_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728
    },
    "gemma3-1b": {
        "model_path": "google/gemma-3-1b-pt",
        "model_type": "gemma3",
        "num_layers": 26,
        "hidden_size": 1152,
        "intermediate_size": 6912
    },
}


# POOLING_STRATEGIES = ["mean", "last"]
POOLING_STRATEGIES = ["mean"]
REPRESENTATION_TYPES = ["residual", "mlp"]
