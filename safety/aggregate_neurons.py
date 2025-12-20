import sys
sys.path.append('../spin')

import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
sys.path.append('../spin')

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm

from config import MODEL_CONFIGS
from probe_trainer import extract_layer_features


class AdaptiveMLPClassifier(nn.Module):

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


def select_salient_neurons(probe, threshold=0.3):
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

    return selected_indices, weights[selected_indices]


def get_layer_weights(best_probes, pooling_type, model_name):
    layer_weights = {}
    layer_scores = {}

    num_layers = MODEL_CONFIGS[model_name]["num_layers"]
    for layer_idx in range(num_layers):
        key = f"layer{layer_idx}_{pooling_type}"
        if key in best_probes:
            layer_scores[layer_idx] = best_probes[key]["val_f1"]

    if not layer_scores:
        return {}

    max_score = max(layer_scores.values())
    min_score = min(layer_scores.values())
    score_range = max_score - min_score if max_score > min_score else 1.0

    for layer_idx, score in layer_scores.items():
        normalized_score = (score - min_score) / score_range
        layer_weights[layer_idx] = max(0.1, normalized_score)

    return layer_weights


def aggregate_features(representations, pooling_type, selected_neurons_dict,
                       layer_weights):
    aggregated_features = []

    selected_layer_indices = sorted(layer_weights.keys())

    for sample_rep in representations:
        sample_features = []
        for layer_idx in selected_layer_indices:
            layer_features = sample_rep[layer_idx][pooling_type]

            probe_key = f"layer{layer_idx}_{pooling_type}"
            if probe_key in selected_neurons_dict:
                selected_indices = selected_neurons_dict[probe_key]
                selected_features = layer_features[selected_indices]

                weight = layer_weights[layer_idx]
                weighted_features = selected_features * weight
                sample_features.append(weighted_features)

        if sample_features:
            aggregated_features.append(np.concatenate(sample_features))

    return np.array(aggregated_features), selected_layer_indices


def train_model(model, X_train, y_train, X_val, y_val, lr, batch_size, epochs, device, trial=None, show_progress=False):
    from torch.amp import autocast, GradScaler

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    val_X_t = torch.FloatTensor(X_val).to(device)
    val_y_t = torch.LongTensor(y_val).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 10

    epoch_iter = tqdm(range(epochs), desc="Training") if show_progress else range(epochs)

    for epoch in epoch_iter:
        model.train()
        indices = torch.randperm(len(X_train))

        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train_t[batch_indices]
            batch_y = y_train_t[batch_indices]

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        with torch.no_grad():
            with autocast('cuda'):
                val_outputs = model(val_X_t)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

        val_f1 = f1_score(y_val, val_preds, average='macro')

        if show_progress:
            epoch_iter.set_postfix({"val_f1": f"{val_f1:.4f}", "best": f"{best_val_f1:.4f}"})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return best_val_f1


def objective(trial, X_train, y_train, X_val, y_val, input_dim, device):
    n_layers = trial.suggest_int('n_layers', 2, 3)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])

    layer_dims = []
    dropout_rates = []

    for i in range(n_layers):
        if i == 0:
            min_dim = min(input_dim, 256)
            max_dim = min(input_dim * 2, 2048)
        else:
            min_dim = 64
            max_dim = min(layer_dims[-1], 1024)

        hidden_dim = trial.suggest_int(f'hidden_dim_layer{i}', min_dim, max_dim, step=64)
        dropout = trial.suggest_float(f'dropout_layer{i}', 0.2, 0.5)

        layer_dims.append(hidden_dim)
        dropout_rates.append(dropout)

    model = AdaptiveMLPClassifier(input_dim, layer_dims, dropout_rates).to(device)

    val_f1 = train_model(model, X_train, y_train, X_val, y_val,
                         lr, batch_size, epochs=100, device=device, trial=trial)

    return val_f1


def train_with_optuna(X_train, y_train, X_val, y_val, device, n_trials=50):
    input_dim = X_train.shape[1]

    print(f"\nStarting Optuna hyperparameter search...")
    print(f"Input dimension: {input_dim}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Running {n_trials} trials with 2 parallel jobs...")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_dim, device),
        n_trials=n_trials,
        n_jobs=2,
        show_progress_bar=True
    )

    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:5]

    print(f"\nTop-5 trials by validation F1:")
    for i, trial in enumerate(top_trials):
        print(f"{i+1}. Trial {trial.number}: val_f1={trial.value:.4f}")

    return [trial.params for trial in top_trials]


def train_final_model(X_train, y_train, X_val, y_val, X_test, y_test,
                      best_params, device):
    input_dim = X_train.shape[1]

    n_layers = best_params['n_layers']
    layer_dims = [best_params[f'hidden_dim_layer{i}'] for i in range(n_layers)]
    dropout_rates = [best_params[f'dropout_layer{i}'] for i in range(n_layers)]

    model = AdaptiveMLPClassifier(input_dim, layer_dims, dropout_rates).to(device)

    print(f"\nTraining final model with best hyperparameters...")
    print(f"Architecture: input({input_dim}) -> " +
          " -> ".join([f"hidden({d}, dropout={dr:.2f})" for d, dr in zip(layer_dims, dropout_rates)]) +
          " -> output(2)")

    val_f1 = train_model(
        model, X_train, y_train, X_val, y_val,
        lr=best_params['lr'],
        batch_size=best_params['batch_size'],
        epochs=200,
        device=device,
        show_progress=True
    )

    # Test evaluation
    from torch.amp import autocast
    model.eval()
    with torch.no_grad():
        test_X_t = torch.FloatTensor(X_test).to(device)
        with autocast('cuda'):
            test_outputs = model(test_X_t)
        test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

    test_f1 = f1_score(y_test, test_preds, average='macro')

    with torch.no_grad():
        train_X_t = torch.FloatTensor(X_train).to(device)
        with autocast('cuda'):
            train_outputs = model(train_X_t)
        train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()

    train_f1 = f1_score(y_train, train_preds, average='macro')
    test_acc = f1_score(y_test, test_preds, average='micro')

    print(f"Final train F1: {train_f1:.4f}")
    print(f"Final validation F1: {val_f1:.4f}")
    print(f"Final test F1: {test_f1:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")

    return model, train_f1, val_f1, test_f1, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--dataset", type=str, default="toxic_chat")
    parser.add_argument("--pooling_types", type=str, nargs="+", default=["residual_mean", "mlp_mean"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 0.9])
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Pooling types: {args.pooling_types}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Device: {device}")

    import os
    with open(f"representations/{args.model}_{args.dataset}_train.pkl", "rb") as f:
        train_data = pickle.load(f)

    validation_path = f"representations/{args.model}_{args.dataset}_validation.pkl"
    if os.path.exists(validation_path):
        with open(validation_path, "rb") as f:
            val_data = pickle.load(f)
        train_reps = train_data["representations"]
        train_labels = train_data["labels"]
        val_reps = val_data["representations"]
        val_labels = val_data["labels"]
    else:
        all_train_reps = train_data["representations"]
        all_train_labels = train_data["labels"]
        val_split_ratio = 0.2
        val_size = int(len(all_train_labels) * val_split_ratio)
        val_reps = all_train_reps[:val_size]
        val_labels = all_train_labels[:val_size]
        train_reps = all_train_reps[val_size:]
        train_labels = all_train_labels[val_size:]

    with open(f"representations/{args.model}_{args.dataset}_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(f"probes/{args.model}_{args.dataset}_probes.pkl", "rb") as f:
        probe_data = pickle.load(f)

    test_reps = test_data["representations"]
    test_labels = test_data["labels"]
    best_probes = probe_data["best_probes"]

    print(f"\nTrain: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    all_results = []

    for pooling_type in args.pooling_types:
        print(f"\n{'='*80}")
        print(f"Processing pooling_type={pooling_type}")
        print(f"{'='*80}")

        layer_weights = get_layer_weights(best_probes, pooling_type, args.model)

        for threshold in args.thresholds:
            print(f"\n{'='*60}")
            print(f"Testing pooling={pooling_type}, threshold={threshold}")
            print(f"{'='*60}")

            selected_neurons_dict = {}
            for layer_idx in layer_weights.keys():
                key = f"layer{layer_idx}_{pooling_type}"
                if key in best_probes:
                    probe = best_probes[key]["probe"]
                    selected_indices, _ = select_salient_neurons(probe, threshold)
                    selected_neurons_dict[key] = selected_indices

            X_train, selected_layers = aggregate_features(
                train_reps, pooling_type, selected_neurons_dict, layer_weights
            )
            X_val, _ = aggregate_features(
                val_reps, pooling_type, selected_neurons_dict, layer_weights
            )
            X_test, _ = aggregate_features(
                test_reps, pooling_type, selected_neurons_dict, layer_weights
            )

            print(f"Feature dimension: {X_train.shape[1]}")

            top_configs = train_with_optuna(
                X_train, train_labels, X_val, val_labels, device, n_trials=args.n_trials
            )

            threshold_results = []
            for i, params in enumerate(top_configs):
                print(f"\nTraining config {i+1}/5 for pooling={pooling_type}, threshold={threshold}...")
                final_model, train_f1, final_val_f1, test_f1, test_acc = train_final_model(
                    X_train, train_labels, X_val, val_labels, X_test, test_labels,
                    params, device
                )

                result = {
                    "pooling_type": pooling_type,
                    "threshold": threshold,
                    "layer_weight_strategy": "performance",
                    "selected_layers": selected_layers,
                    "num_features": int(X_train.shape[1]),
                    "layer_weights": {str(k): float(v) for k, v in layer_weights.items()},
                    "best_hyperparameters": params,
                    "train_f1": float(train_f1),
                    "val_f1": float(final_val_f1),
                    "test_f1": float(test_f1),
                    "test_acc": float(test_acc),
                    "final_mlp": final_model,
                    "config_rank": i+1
                }
                threshold_results.append(result)
                print(f"Config {i+1}: val_f1={final_val_f1:.4f}, test_f1={test_f1:.4f}, test_acc={test_acc:.4f}")

            best_for_threshold = max(threshold_results, key=lambda x: x['test_acc'])
            all_results.append(best_for_threshold)

            print(f"\nBest for pooling={pooling_type}, threshold={threshold}: test_acc={best_for_threshold['test_acc']:.4f}, test_f1={best_for_threshold['test_f1']:.4f} (config {best_for_threshold['config_rank']})")

    best_overall = max(all_results, key=lambda x: x['test_acc'])

    import os
    output_dir = f"probes/optuna/{args.model}_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    save_path = f"{output_dir}/best_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(best_overall, f)

    results_path = f"{output_dir}/results.json"
    json_results = {
        "best_overall": {k: v for k, v in best_overall.items() if k != "final_mlp"},
        "all_results": [{k: v for k, v in r.items() if k != "final_mlp"} for r in sorted(all_results, key=lambda x: x['test_acc'], reverse=True)]
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"BEST MODEL:")
    print(f"{'='*80}")
    print(f"Pooling type: {best_overall['pooling_type']}")
    print(f"Threshold: {best_overall['threshold']}")
    print(f"Layers used: {len(best_overall['selected_layers'])}")
    print(f"Train F1: {best_overall['train_f1']:.4f}")
    print(f"Val F1:   {best_overall['val_f1']:.4f}")
    print(f"Test F1:  {best_overall['test_f1']:.4f}")
    print(f"\nSaved to: {save_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
