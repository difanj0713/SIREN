"""
SPIN Per-Token Analysis
Process a single sentence through a trained SPIN checkpoint and visualize token-level scores.
"""

import sys
import os
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, os.path.abspath('../../../spin'))
from model_hooks import Qwen3RepresentationExtractor

sys.path.insert(0, os.path.abspath('../../'))
from config import MODEL_CONFIGS

import torch.nn as nn


class LinearProbe:
    """
    Linear probe class needed for loading pickled probe results.
    """
    def get_feature_importance(self):
        # This is a stub - actual implementation would be in the loaded probe
        pass


class AdaptiveMLPClassifier(nn.Module):
    """
    MLP classifier used in SPIN checkpoints.
    This class is needed for loading pickled SPIN checkpoints.
    """
    def __init__(self, input_dim, layer_dims, dropout_rates, num_classes=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, (hidden_dim, dropout) in enumerate(zip(layer_dims, dropout_rates)):
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        final_linear = nn.Linear(prev_dim, num_classes)
        nn.init.kaiming_normal_(final_linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def select_salient_neurons(probe, threshold):
    """
    Select salient neurons based on feature importance and threshold.
    Same logic as in evaluate_robustness_spin.py
    """
    weights = probe.get_feature_importance()
    total_importance = np.sum(weights)
    sorted_indices = np.argsort(weights)[::-1]
    selected_indices = []
    cumulative_importance = 0.0
    for idx in sorted_indices:
        selected_indices.append(idx)
        cumulative_importance += weights[idx]
        if cumulative_importance >= threshold * total_importance:
            break
    return selected_indices


def load_spin_checkpoint(checkpoint_path, probe_results_path=None, model_name=None):
    """
    Load SPIN checkpoint from a pickle file.
    
    Args:
        checkpoint_path: Path to the SPIN checkpoint pickle file
        probe_results_path: Optional path to probe results pickle file (needed if selected_neurons_dict not in checkpoint)
        model_name: Model name (e.g., 'qwen3-0.6b') - needed to get num_layers from MODEL_CONFIGS
    
    Returns:
        Loaded SPIN checkpoint dictionary with selected_neurons_dict computed if needed
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SPIN checkpoint not found at: {checkpoint_path}")
    
    print(f"Loading SPIN checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded successfully")
    print(f"  Pooling type: {checkpoint.get('pooling_type', 'N/A')}")
    print(f"  Threshold: {checkpoint.get('threshold', 'N/A')}")
    print(f"  Selected layers: {checkpoint.get('selected_layers', 'N/A')}")
    
    # Check if selected_neurons_dict exists in checkpoint
    if "selected_neurons_dict" not in checkpoint:
        print("  selected_neurons_dict not found in checkpoint")
        if probe_results_path is None:
            raise ValueError(
                "selected_neurons_dict not in checkpoint. Please provide --probe_results_path "
                "to load probe results and compute selected neurons."
            )
        
        print(f"Loading probe results from: {probe_results_path}")
        if not os.path.exists(probe_results_path):
            raise FileNotFoundError(f"Probe results not found at: {probe_results_path}")
        
        # Check if file is JSON or pickle
        file_ext = os.path.splitext(probe_results_path)[1].lower()
        if file_ext == '.json':
            raise ValueError(
                f"Probe results file must be a pickle file (.pkl), not JSON.\n"
                f"JSON files don't contain the probe objects needed for neuron selection.\n"
                f"Please provide the probe results pickle file (e.g., model_dataset_probes.pkl)\n"
                f"that contains 'best_probes' with probe objects."
            )
        
        # Load pickle file
        try:
            with open(probe_results_path, 'rb') as f:
                probe_results = pickle.load(f)
        except Exception as e:
            raise ValueError(
                f"Failed to load probe results from {probe_results_path}.\n"
                f"Error: {e}\n"
                f"Please ensure this is a valid pickle file containing 'best_probes'."
            )
        
        if 'best_probes' not in probe_results:
            raise ValueError(
                f"Probe results file does not contain 'best_probes' key.\n"
                f"Expected structure: {{'best_probes': {{...}}}}\n"
                f"Found keys: {list(probe_results.keys())}"
            )
        
        best_probes = probe_results['best_probes']
        pooling_type = checkpoint["pooling_type"]
        threshold = checkpoint["threshold"]
        
        # Get num_layers from MODEL_CONFIGS (same as evaluate_robustness_spin.py)
        if model_name is None:
            model_name = checkpoint.get('model', 'qwen3-0.6b')
        
        sys.path.insert(0, os.path.abspath('../../'))
        from config import MODEL_CONFIGS
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS. Available models: {list(MODEL_CONFIGS.keys())}")
        
        num_layers = MODEL_CONFIGS[model_name]["num_layers"]
        
        # Compute selected_neurons_dict from probe results (same as evaluate_robustness_spin.py)
        # Iterate through ALL layers (0 to num_layers-1), not just selected_layers
        print(f"Computing selected_neurons_dict from probe results (checking {num_layers} layers)...")
        selected_neurons_dict = {}
        for layer_idx in range(num_layers):
            key = f"layer{layer_idx}_{pooling_type}"
            if key in best_probes:
                probe = best_probes[key]["probe"]
                selected_indices = select_salient_neurons(probe, threshold)
                selected_neurons_dict[key] = selected_indices
        
        checkpoint["selected_neurons_dict"] = selected_neurons_dict
        print(f"  Computed selected_neurons_dict for {len(selected_neurons_dict)} layers")
    else:
        print(f"  selected_neurons_dict found in checkpoint ({len(checkpoint['selected_neurons_dict'])} layers)")
    
    return checkpoint


def append_average_representation(representation):
    """
    Append average (mean-pooled) representation as an additional token position.
    This allows computing the average score using the same per-token pipeline.
    
    Args:
        representation: Dictionary with layer_idx -> dict with rep_type -> numpy array of shape [seq_len, hidden_dim]
    
    Returns:
        Modified representation with average appended (seq_len becomes seq_len + 1)
    """
    # Compute average for each layer and representation type
    for layer_idx in representation:
        for rep_type in representation[layer_idx]:
            # representation[layer_idx][rep_type] is [seq_len, hidden_dim]
            layer_rep = representation[layer_idx][rep_type]
            if layer_rep.shape[0] > 0:
                # Compute mean across sequence length dimension
                avg_rep = np.mean(layer_rep, axis=0, keepdims=True)  # [1, hidden_dim]
                # Append to the sequence
                representation[layer_idx][rep_type] = np.vstack([layer_rep, avg_rep])  # [seq_len+1, hidden_dim]
    
    return representation


def extract_per_token_representations(text, model_name, model_configs, include_average=True):
    """
    Extract per-token representations from the base model using hooks.
    
    Args:
        text: Single text string to process
        model_name: Name of the model (for config lookup)
        model_configs: Model configuration dictionary
        include_average: If True, append average representation as an additional token position
    
    Returns:
        Dictionary with layer_idx -> dict with rep_type -> numpy array of shape [seq_len, hidden_dim]
        If include_average=True, seq_len includes one extra position for the average
    """
    model_config = model_configs[model_name]
    
    print(f"Initializing representation extractor for model: {model_name}")
    extractor = Qwen3RepresentationExtractor(
        model_config["model_path"],
        device="cuda",
        batch_size=1,
        rep_types=["residual", "mlp"]  # Raw per-token representations
    )
    extractor.register_hooks()
    
    print(f"Extracting per-token representations...")
    with torch.no_grad():
        batch_reps = extractor.extract_batch([text])
    
    extractor.remove_hooks()
    del extractor
    torch.cuda.empty_cache()
    
    # Get the first (and only) representation
    representation = batch_reps[0] if batch_reps else {}
    
    # Append average representation as an additional token position
    if include_average and representation:
        print("Appending average representation as additional token position...")
        representation = append_average_representation(representation)
    
    return representation


def aggregate_per_token_features(representation, pooling_type, selected_neurons_dict, layer_weights, selected_layers):
    """
    Aggregate per-token features across selected layers and neurons.
    
    Args:
        representation: Single sample representation dict (layer_idx -> rep_type -> [seq_len, hidden_dim])
        pooling_type: Pooling type (e.g., "residual_mean" or "mlp_mean")
        selected_neurons_dict: Dictionary mapping layer keys to selected neuron indices
        layer_weights: Dictionary of layer weights
        selected_layers: List of selected layer indices
    
    Returns:
        Numpy array of shape [seq_len, aggregated_feature_dim]
    """
    # Determine sequence length
    seq_len = None
    for layer_idx in selected_layers:
        key = f"layer{layer_idx}_{pooling_type}"
        if key in selected_neurons_dict and layer_idx in representation:
            # Map pooling_type to raw representation type
            rep_type = "residual" if pooling_type == "residual_mean" else "mlp"
            if rep_type in representation[layer_idx]:
                seq_len = representation[layer_idx][rep_type].shape[0]
                break
    
    if seq_len is None:
        raise ValueError("Could not determine sequence length from representations")
    
    # Aggregate features for each token position
    token_features = []
    for token_idx in range(seq_len):
        sample_features = []
        for layer_idx in selected_layers:
            key = f"layer{layer_idx}_{pooling_type}"
            if key not in selected_neurons_dict or layer_idx not in representation:
                continue
            
            # Map pooling_type to raw representation type
            rep_type = "residual" if pooling_type == "residual_mean" else "mlp"
            if rep_type not in representation[layer_idx]:
                continue
            
            # Get per-token representation for this token position
            layer_features = representation[layer_idx][rep_type]  # [seq_len, hidden_dim]
            if token_idx >= layer_features.shape[0]:
                continue
            
            token_layer_features = layer_features[token_idx]  # [hidden_dim]
            selected_indices = selected_neurons_dict[key]
            selected_features = token_layer_features[selected_indices]  # [num_selected]
            weight = layer_weights[str(layer_idx)]
            weighted_features = selected_features * weight
            sample_features.append(weighted_features)
        
        if sample_features:
            token_features.append(np.concatenate(sample_features))
        else:
            token_features.append(np.array([]))
    
    return np.array(token_features)  # [seq_len, aggregated_feature_dim]


def get_per_token_spin_scores(representation, spin_checkpoint):
    """
    Get per-token SPIN scores (probabilities) for a single sentence.
    
    Args:
        representation: Single sample representation dictionary
        spin_checkpoint: Loaded SPIN checkpoint dictionary
    
    Returns:
        Numpy array of probability scores for class 1 (detected) for each token position
    """
    pooling_type = spin_checkpoint["pooling_type"]
    selected_neurons_dict = spin_checkpoint["selected_neurons_dict"]
    layer_weights = spin_checkpoint["layer_weights"]
    selected_layers = spin_checkpoint["selected_layers"]
    
    # Aggregate features for all tokens
    X = aggregate_per_token_features(
        representation, pooling_type, selected_neurons_dict, 
        layer_weights, selected_layers
    )  # [seq_len, aggregated_feature_dim]
    
    if X.shape[0] == 0:
        return np.array([])
    
    # Get SPIN model
    model = spin_checkpoint["final_mlp"]
    device = next(model.parameters()).device
    model.eval()
    
    # Process all tokens
    token_scores = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
            outputs = model(batch_X)
            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            # Get probability of class 1 (detected)
            class1_probs = probs[:, 1].cpu().numpy()
            token_scores.extend(class1_probs)
    
    return np.array(token_scores)


def tokenize_text(text, model_name, model_configs, include_average=True):
    """
    Tokenize text and return tokens for visualization.
    
    Args:
        text: Input text string
        model_name: Name of the model (for config lookup)
        model_configs: Model configuration dictionary
        include_average: If True, append a special token for the average score
    
    Returns:
        List of token strings (with [AVG] token appended if include_average=True)
    """
    model_config = model_configs[model_name]
    
    # Create extractor to use tokenizer
    extractor = Qwen3RepresentationExtractor(
        model_config["model_path"],
        device="cuda",
        batch_size=1,
        rep_types=["residual_mean", "mlp_mean"]
    )
    
    # Tokenize
    text_clean = text.strip() if text.strip() else " "
    inputs = extractor.tokenizer(
        text_clean,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Get token strings
    token_ids = inputs["input_ids"][0].cpu().numpy()
    attention_mask = inputs["attention_mask"][0].cpu().numpy()
    token_strings = extractor.tokenizer.convert_ids_to_tokens(token_ids)
    
    tokens = []
    for token_str, mask_val in zip(token_strings, attention_mask):
        if mask_val == 0:  # Skip padding tokens
            break
        # Replace special tokens
        if token_str == extractor.tokenizer.unk_token:
            token_str = "[UNK]"
        elif token_str == extractor.tokenizer.pad_token:
            token_str = "[PAD]"
        elif token_str == extractor.tokenizer.eos_token:
            token_str = "[EOS]"
        elif token_str == extractor.tokenizer.bos_token:
            token_str = "[BOS]"
        # Clean up subword prefixes
        if token_str.startswith('##'):
            token_str = token_str[2:]
        elif token_str.startswith('Ġ'):
            token_str = ' ' + token_str[1:]
        tokens.append(token_str)
    
    # Append special token for average score
    if include_average:
        tokens.append("[AVG]")
    
    del extractor
    torch.cuda.empty_cache()
    
    return tokens


def generate_html_visualization(text, tokens, scores, output_path, display_text=None):
    """
    Generate HTML visualization with tokens colored by SPIN scores.
    
    Args:
        text: Original text (used for processing)
        tokens: List of token strings
        scores: Array of SPIN scores (probabilities) for each token
        output_path: Path to save HTML file
        display_text: Optional text to display instead of original text (shown in plaintext)
    """
    if len(scores) == 0:
        scores = np.zeros(len(tokens))
    
    # Use display_text if provided, otherwise use original text
    text_to_display = display_text if display_text is not None else text
    text_escaped = text_to_display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SPIN Token-Level Analysis</title>
    <style>
        body {{
            font-family: Harding Text Web, Times New Roman, Helvetica, sans-serif;
            max-width: 1200px;
            margin: 0px auto;
            padding: 0px;
            background-color: transparent;
        }}
        .container {{
            background-color: transparent;
            padding: 0px;
        }}
        .display-text {{
            margin: -5px 0px;
            padding: 0px;
            font-size: 16px;
            color: black;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .tokenized-text {{
            margin: -5px 0px;
            line-height: 1.5;
            font-size: 16px;
            padding: 0px;
            background-color: transparent;
            border-radius: 4px;
        }}
        .token {{
            display: inline-block;
            margin: 2px 0px;
            padding: 0px 2px;
            transition: all 0.2s;
        }}
        .token:hover {{
            transform: scale(1.1);
            z-index: 10;
            position: relative;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .legend {{
            margin: 30px 0px;
        }}
        .gradient-bar {{
            width: 100%;
            height: 20px;
            background: linear-gradient(to right, #80ff80 0%, #ffff80 50%, #ff8080 100%);
            position: relative;
        }}
        .gradient-labels {{
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 12px;
            color: black;
        }}
        .avg-token {{
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="display-text">
{f"[Original Text] {text_escaped}"}
        </div>
        
        <div class="tokenized-text" id="tokenized-text">
{f"[Rewritten Text]"}
"""
    
    # Add tokens with colored backgrounds
    for i, (token, score) in enumerate(zip(tokens, scores)):
        intensity = float(score)
        
        # Create color gradient: green (low) -> yellow (medium) -> red (high)
        if intensity < 0.5:
            r = int(255 * (intensity * 2))
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (1 - (intensity - 0.5) * 2))
            b = 0
        
        # Convert to hex
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        
        # Calculate background color (lighter version for readability)
        bg_r = min(255, r + int((255 - r) * 0.5))
        bg_g = min(255, g + int((255 - g) * 0.5))
        bg_b = min(255, b + int((255 - b) * 0.5))
        bg_color = f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}"
        
        # Escape HTML special characters
        token_escaped = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        
        # Special styling for [AVG] token - explicitly display the score
        if token == "[AVG]":
            token_class = "token avg-token"
            title = f"Average SPIN Score: {score:.3f}"
            # Display score explicitly in the token
            token_display = f"[AVG: {score:.3f}]"
            token_escaped = token_display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        else:
            token_class = "token"
            title = f"SPIN Score: {score:.3f}"
        
        html_content += f'            <span class="{token_class}" style="background-color: {bg_color};" title="{title}">'
        html_content += f'{token_escaped}</span>\n'
    
    html_content += f"""        </div>
        
        <div class="legend">
            <div class="gradient-bar"></div>
            <div class="gradient-labels">
                <span>0.0 (Low)</span>
                <span>1.0 (High)</span>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    # Save HTML file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a single sentence with SPIN per-token scores using a trained checkpoint'
    )
    parser.add_argument('--text', type=str, required=True, 
                        help='Input text sentence to analyze')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SPIN checkpoint pickle file (e.g., best_model.pkl)')
    parser.add_argument('--probe', type=str, default=None,
                        help='Path to probe results pickle file (required if selected_neurons_dict not in checkpoint)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., qwen3-0.6b) - must match MODEL_CONFIGS')
    parser.add_argument('--output', type=str, default='results/spin_analysis.html',
                        help='Output HTML file path (default: results/spin_analysis.html)')
    parser.add_argument('--display-text', type=str, default=None,
                        help='Optional text to display instead of original text (shown in plaintext)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SPIN Per-Token Analysis")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Input text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
    print(f"Checkpoint: {args.checkpoint}")
    if args.probe:
        print(f"Probe results: {args.probe}")
    print()
    
    # Step 1: Load SPIN checkpoint
    spin_checkpoint = load_spin_checkpoint(args.checkpoint, args.probe, args.model)
    
    # Step 2: Extract per-token representations from base model (with average appended)
    representation = extract_per_token_representations(args.text, args.model, MODEL_CONFIGS, include_average=True)
    
    # Step 3: Get per-token SPIN scores (includes average score as last token)
    print(f"\nComputing per-token SPIN scores...")
    scores = get_per_token_spin_scores(representation, spin_checkpoint)
    
    if len(scores) == 0:
        print("Error: No scores generated. Check input text and model configuration.")
        return
    
    # Step 4: Tokenize for visualization (with [AVG] token appended)
    print(f"Tokenizing text for visualization...")
    tokens = tokenize_text(args.text, args.model, MODEL_CONFIGS, include_average=True)
    
    # Align tokens and scores (should match since both include average)
    if len(tokens) != len(scores):
        print(f"Warning: Token count ({len(tokens)}) doesn't match score count ({len(scores)})")
        min_len = min(len(tokens), len(scores))
        tokens = tokens[:min_len]
        scores = scores[:min_len]
    
    # Extract average score (last element) and regular token scores
    avg_score = scores[-1] if len(scores) > 0 else 0.0
    token_scores = scores[:-1] if len(scores) > 1 else scores
    
    # Step 5: Generate HTML visualization
    print(f"Generating HTML visualization...")
    generate_html_visualization(args.text, tokens, scores, args.output, args.display_text)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("Analysis Complete")
    print(f"{'=' * 70}")
    print(f"Average SPIN Score (computed): {avg_score:.3f}")
    if len(token_scores) > 0:
        print(f"Average SPIN Score (from tokens): {np.mean(token_scores):.3f}")
        print(f"Max Token Score: {np.max(token_scores):.3f}")
        print(f"Min Token Score: {np.min(token_scores):.3f}")
    print(f"Number of Tokens: {len(tokens) - 1} (plus 1 average)")
    print(f"HTML saved to: {args.output}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
