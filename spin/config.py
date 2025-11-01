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
        "num_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072
    }
}

POOLING_STRATEGIES = ["mean", "last"]
REPRESENTATION_TYPES = ["residual", "mlp"]
