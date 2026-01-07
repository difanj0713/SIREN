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
        "model_path": "Qwen/Qwen3-4B",
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
    "llama3.2-1b": {
        "model_path": "meta-llama/Llama-3.2-1B",
        "model_type": "llama",
        "num_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192
    },
    "llama3.2-3b": {
        "model_path": "meta-llama/Llama-3.2-3B",
        "model_type": "llama",
        "num_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192
    },
    "llama3guard-1b": {
        "model_path": "meta-llama/Llama-Guard-3-1B",
        "model_type": "llama",
        "num_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192
    },
    "llama3guard-8b": {
        "model_path": "meta-llama/Llama-Guard-3-8B",
        "model_type": "llama",
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336
    },
    "llama3.1-8b": {
        "model_path": "meta-llama/Llama-3.1-8B",
        "model_type": "llama",
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336
    },
}

POOLING_STRATEGIES = ["mean"]
REPRESENTATION_TYPES = ["residual", "mlp"]
