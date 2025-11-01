from datasets import load_dataset

# Predefined dataset configurations
DATASET_CONFIGS = {
    "imdb": {
        "hf_name": "stanfordnlp/imdb",
        "hf_config": None,
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "task_description": "sentiment analysis",
        "prompt_template": """You are a sentiment analysis expert. Analyze the following movie review and determine its sentiment.

Review: {text}

Let me analyze this step by step:

1. Language analysis: I'll examine the specific words and phrases used and their emotional valence
2. Opinion indicators: I'll identify explicit statements of opinion, satisfaction, or dissatisfaction
3. Overall tone: I'll assess the general tone and emotional direction of the review
4. Sentiment indicators: I'll look for clear positive or negative sentiment markers

Reasoning:
[Please provide detailed step-by-step analysis here]

Final Classification: ##POSITIVE or ##NEGATIVE""",
        "stop_words": ["##POSITIVE", "##NEGATIVE"],
        "judge_labels": ["positive", "negative"],
        "judge_prompt": """You are an expert judge tasked with determining the sentiment classification from a model's response.

The model was asked to classify a movie review as either POSITIVE or NEGATIVE.

Model's response:
{response}

Based on the model's response above, what is the final sentiment classification?
You must respond with EXACTLY one word: either "positive" or "negative" (lowercase, no punctuation)."""
    },

    "tweet_hate": {
        "hf_name": "tweet_eval",
        "hf_config": "hate",
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "not_hate", 1: "hate"},
        "task_description": "hate speech detection",
        "prompt_template": """You are a content moderation expert. Analyze the following tweet and determine if it contains hate speech.

Tweet: {text}

Reminder: {spin_verbalization}

Analyze this step by step and please provide detailed step-by-step analysis here.

Final Classification: ##HATE or ##NOT_HATE""",
        "stop_words": ["##HATE", "##NOT_HATE"],
        "judge_labels": ["hate", "not_hate"],
        "judge_prompt": """You are an expert judge tasked with determining the hate speech classification from a model's response.

You must respond with EXACTLY one of these: "hate" or "not_hate" (lowercase, no punctuation)."""
    },

    "ag_news": {
        "hf_name": "ag_news",
        "hf_config": None,
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "world", 1: "sports", 2: "business", 3: "sci_tech"},
        "task_description": "news topic classification",
        "prompt_template": """You are a news categorization expert. Analyze the following news article and determine its topic category.

Article: {text}

Let me analyze this step by step:

1. Content analysis: I'll examine the main subject matter and key topics discussed
2. Domain indicators: I'll identify specific terminology, entities, or concepts that indicate the news domain
3. Context clues: I'll consider the overall context and focus of the article
4. Category matching: I'll match the content to the appropriate news category

Reasoning:
[Please provide detailed step-by-step analysis here]

Final Classification: ##WORLD, ##SPORTS, ##BUSINESS, or ##SCI_TECH""",
        "stop_words": ["##WORLD", "##SPORTS", "##BUSINESS", "##SCI_TECH"],
        "judge_labels": ["world", "sports", "business", "sci_tech"],
        "judge_prompt": """You are an expert judge tasked with determining the topic classification from a model's response.

The model was asked to classify a news article into one of: WORLD, SPORTS, BUSINESS, or SCI/TECH.

Model's response:
{response}

Based on the model's response above, what is the final topic classification?
You must respond with EXACTLY one of these: "world", "sports", "business", or "sci_tech" (lowercase, no punctuation)."""
    }
}

def load_test_data(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    if config["hf_config"]:
        dataset = load_dataset(config["hf_name"], config["hf_config"])
    else:
        dataset = load_dataset(config["hf_name"])

    test_samples = dataset[config["test_split"]]

    samples = []
    for idx, item in enumerate(test_samples):
        text = item[config["text_field"]]
        label_idx = item[config["label_field"]]
        label = config["label_map"][label_idx]

        samples.append({
            "id": idx,
            "text": text,
            "label": label
        })

    return samples, config

def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]
