# Predefined model configurations
MODEL_CONFIGS = {
    "qwen3-0.6b": {
        "model_path": "Qwen/Qwen3-0.6B",
        "use_chat_template": True,
        "enable_thinking": False,
        "supports_system": True
    },
    "qwen3-1.5b": {
        "model_path": "Qwen/Qwen3-1.7B",
        "use_chat_template": True,
        "enable_thinking": False,
        "supports_system": True
    },
    "qwen2.5-7b": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "use_chat_template": True,
        "enable_thinking": False,
        "supports_system": True
    },
    "llama3-8b": {
        "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "use_chat_template": True,
        "enable_thinking": False,
        "supports_system": True
    },
    "mistral-7b": {
        "model_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "use_chat_template": True,
        "enable_thinking": False,
        "supports_system": False
    }
}

def get_model_config(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def format_prompt(tokenizer, model_config, user_prompt):
    if model_config["use_chat_template"]:
        messages = [{"role": "user", "content": user_prompt}]

        apply_chat_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True
        }

        if "enable_thinking" in model_config:
            apply_chat_kwargs["enable_thinking"] = model_config["enable_thinking"]

        prompt = tokenizer.apply_chat_template(messages, **apply_chat_kwargs)
        return prompt
    else:
        return user_prompt
