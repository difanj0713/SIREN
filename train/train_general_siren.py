import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from utils.config import MODEL_CONFIGS
from utils.model_hooks import Qwen3RepresentationExtractor
from train.preprocess import preprocess_dataset
from train.probe_trainer import train_and_evaluate_probe, extract_layer_features


def compute_per_dataset_f1(y_true, y_pred, dataset_ids):
    unique_datasets = np.unique(dataset_ids)
    dataset_f1s = []
    for dataset_id in unique_datasets:
        mask = dataset_ids == dataset_id
        dataset_f1 = f1_score(y_true[mask], y_pred[mask], average='macro')
        dataset_f1s.append(dataset_f1)
    return np.mean(dataset_f1s)


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

def extract_all_representations(model_name, datasets, device, batch_size, rep_types, val_ratio=0.2):
    model_config = MODEL_CONFIGS[model_name]
    extractor = Qwen3RepresentationExtractor(
        model_config["model_path"],
        device=device,
        batch_size=batch_size,
        rep_types=rep_types
    )
    extractor.register_hooks()

    all_reps = {}
    for split_name in ["train", "validation", "test"]:
        print(f"\nProcessing {split_name} split...")
        all_texts = []
        all_labels = []
        all_dataset_ids = []

        for dataset_idx, dataset_name in enumerate(datasets):
            dataset = preprocess_dataset(dataset_name, val_ratio)
            split_data = dataset[split_name]
            all_texts.extend([item["text"] for item in split_data])
            all_labels.extend([item["label"] for item in split_data])
            all_dataset_ids.extend([dataset_idx] * len(split_data))

        indices = np.random.RandomState(42).permutation(len(all_texts))
        all_texts = [all_texts[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        all_dataset_ids = [all_dataset_ids[i] for i in indices]

        print(f"Total samples in {split_name}: {len(all_labels)}")

        representations = []
        for i in tqdm(range(0, len(all_texts), batch_size), desc=f"Extracting {split_name}"):
            batch_texts = all_texts[i:i+batch_size]
            with torch.no_grad():
                batch_reps = extractor.extract_batch(batch_texts)
                representations.extend(batch_reps)
                torch.cuda.empty_cache()

        all_reps[split_name] = {
            "representations": representations,
            "labels": np.array(all_labels),
            "dataset_ids": np.array(all_dataset_ids),
            "num_layers": model_config["num_layers"]
        }

    extractor.remove_hooks()
    del extractor
    torch.cuda.empty_cache()
    return all_reps

def train_probes(train_reps, train_labels, val_reps, val_labels, val_dataset_ids, test_reps, test_labels, test_dataset_ids, num_layers, c_values, rep_types, device):
    best_probes = {}

    for pooling_type in rep_types:
        print(f"\nTraining {pooling_type} probes...")
        for layer_idx in range(num_layers):
            rep_type, pooling = pooling_type.split("_")[0], "_".join(pooling_type.split("_")[1:])

            probe, train_f1, val_f1, best_C = train_and_evaluate_probe(
                train_reps, train_labels, val_reps, val_labels, val_dataset_ids,
                layer_idx, rep_type, pooling, c_values, device, metric='f1_macro'
            )

            test_X = extract_layer_features(test_reps, layer_idx, rep_type, pooling)
            test_f1 = probe.evaluate(test_X, test_labels, test_dataset_ids, metric='f1_macro')

            key = f"layer{layer_idx}_{pooling_type}"
            best_probes[key] = {
                "layer": layer_idx,
                "rep_type": rep_type,
                "pooling": pooling,
                "best_C": best_C,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "test_f1": test_f1,
                "probe": probe
            }
            print(f"  Layer {layer_idx:2d}: Train_F1={train_f1:.4f} Val_F1={val_f1:.4f} Test_F1={test_f1:.4f} C={best_C}")

    return best_probes

def select_salient_neurons(probe, threshold):
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

def get_layer_weights(best_probes, pooling_type, num_layers):
    layer_scores = {}
    for layer_idx in range(num_layers):
        key = f"layer{layer_idx}_{pooling_type}"
        if key in best_probes:
            layer_scores[layer_idx] = best_probes[key]["val_f1"]

    if not layer_scores:
        return {}

    max_score = max(layer_scores.values())
    min_score = min(layer_scores.values())
    score_range = max_score - min_score if max_score > min_score else 1.0

    layer_weights = {}
    for layer_idx, score in layer_scores.items():
        normalized_score = (score - min_score) / score_range
        layer_weights[layer_idx] = max(0.1, normalized_score)
    return layer_weights

def aggregate_features(representations, pooling_type, selected_neurons_dict, layer_weights, selected_layers=None):
    if selected_layers is None:
        selected_layers = sorted(layer_weights.keys())
    aggregated_features = []
    for sample_rep in representations:
        sample_features = []
        for layer_idx in selected_layers:
            key = f"layer{layer_idx}_{pooling_type}"
            if key not in selected_neurons_dict:
                continue
            layer_features = sample_rep[layer_idx][pooling_type]
            selected_indices = selected_neurons_dict[key]
            selected_features = layer_features[selected_indices]
            weight = layer_weights[layer_idx]
            weighted_features = selected_features * weight
            sample_features.append(weighted_features)
        if sample_features:
            aggregated_features.append(np.concatenate(sample_features))
    return np.array(aggregated_features), selected_layers

def train_model(model, X_train, y_train, X_val, y_val, val_dataset_ids, lr, batch_size, epochs, device, trial=None, show_progress=False, use_gpu_data=False):
    from torch.amp import autocast, GradScaler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 10
    epoch_iter = tqdm(range(epochs), desc="Training") if show_progress else range(epochs)

    if use_gpu_data and device.type == 'cuda':
        X_train_gpu = torch.FloatTensor(X_train).to(device)
        y_train_gpu = torch.LongTensor(y_train).to(device)
        X_val_gpu = torch.FloatTensor(X_val).to(device) if len(X_val) > 0 else None
    else:
        X_train_gpu = None
        y_train_gpu = None
        X_val_gpu = None

    for epoch in epoch_iter:
        model.train()
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            if X_train_gpu is not None:
                batch_X = X_train_gpu[batch_indices]
                batch_y = y_train_gpu[batch_indices]
            else:
                batch_X = torch.FloatTensor(X_train[batch_indices]).to(device)
                batch_y = torch.LongTensor(y_train[batch_indices]).to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        all_val_preds = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                if X_val_gpu is not None:
                    batch_X = X_val_gpu[i:i+batch_size]
                else:
                    batch_X = torch.FloatTensor(X_val[i:i+batch_size]).to(device)
                with autocast('cuda'):
                    val_outputs = model(batch_X)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                all_val_preds.extend(val_preds)
        val_preds = np.array(all_val_preds)
        val_f1 = compute_per_dataset_f1(y_val, val_preds, val_dataset_ids)

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
            if show_progress:
                print(f"\nEarly stopping at epoch {epoch+1}/{epochs}")
            break
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    return best_val_f1

def objective(trial, X_train, y_train, train_dataset_ids, input_dim, device, n_folds=5, use_gpu_data=False):
    n_layers = trial.suggest_int('n_layers', 2, 3)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = 8192
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

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_val_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        fold_val_dataset_ids = train_dataset_ids[val_idx]

        model = AdaptiveMLPClassifier(input_dim, layer_dims, dropout_rates).to(device)
        fold_val_f1 = train_model(model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, fold_val_dataset_ids, lr, batch_size, epochs=100, device=device, trial=None, use_gpu_data=use_gpu_data)
        fold_val_f1s.append(fold_val_f1)
        del model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, fold_val_dataset_ids
        torch.cuda.empty_cache()

    mean_val_f1 = np.mean(fold_val_f1s)
    return mean_val_f1

def train_with_optuna(X_train, y_train, train_dataset_ids, X_val, y_val, val_dataset_ids, device, n_trials, n_jobs, n_folds=5, use_cv=True, use_gpu_data=False):
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    dataset_ids_full = np.concatenate([train_dataset_ids, val_dataset_ids])

    input_dim = X_full.shape[1]
    print(f"\nStarting Optuna hyperparameter search with {f'{n_folds}-fold CV' if use_cv else 'fixed split'}...")
    print(f"Input dimension: {input_dim}")

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42), pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20))
    if use_cv:
        study.optimize(lambda trial: objective(trial, X_full, y_full, dataset_ids_full, input_dim, device, n_folds=n_folds, use_gpu_data=use_gpu_data), n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    else:
        study.optimize(lambda trial: objective(trial, X_train, y_train, train_dataset_ids, input_dim, device, n_folds=1, use_gpu_data=use_gpu_data), n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best_trial = max(study.trials, key=lambda t: t.value if t.value is not None else -1)
    print(f"\nBest trial by {'CV' if use_cv else 'validation'} F1:")
    print(f"Trial {best_trial.number}: {'cv' if use_cv else 'val'}_f1={best_trial.value:.4f}")
    return best_trial.params, best_trial.value

def train_final_model(X_train, y_train, X_val, y_val, val_dataset_ids, X_test, y_test, test_dataset_ids, best_params, device, use_val=True, cv_f1=None, use_gpu_data=False):
    from torch.amp import autocast
    input_dim = X_train.shape[1]
    n_layers = best_params['n_layers']
    layer_dims = [best_params[f'hidden_dim_layer{i}'] for i in range(n_layers)]
    dropout_rates = [best_params[f'dropout_layer{i}'] for i in range(n_layers)]
    model = AdaptiveMLPClassifier(input_dim, layer_dims, dropout_rates).to(device)
    print(f"\nTraining final model with best hyperparameters...")
    batch_size = 8192
    if use_val and len(X_val) > 0:
        val_f1 = train_model(model, X_train, y_train, X_val, y_val, val_dataset_ids, lr=best_params['lr'], batch_size=batch_size, epochs=256, device=device, show_progress=True, use_gpu_data=use_gpu_data)
    else:
        from torch.amp import GradScaler
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler('cuda')
        epoch_iter = tqdm(range(512), desc="Training")
        for epoch in epoch_iter:
            model.train()
            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = torch.FloatTensor(X_train[batch_indices]).to(device)
                batch_y = torch.LongTensor(y_train[batch_indices]).to(device)
                optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        val_f1 = cv_f1 if cv_f1 is not None else 0.0

    model.eval()
    all_test_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 2048):
            batch_X = torch.FloatTensor(X_test[i:i+2048]).to(device)
            with autocast('cuda'):
                test_outputs = model(batch_X)
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            all_test_preds.extend(test_preds)
    test_preds = np.array(all_test_preds)
    test_f1 = compute_per_dataset_f1(y_test, test_preds, test_dataset_ids)

    all_train_preds = []
    with torch.no_grad():
        for i in range(0, len(X_train), 2048):
            batch_X = torch.FloatTensor(X_train[i:i+2048]).to(device)
            with autocast('cuda'):
                train_outputs = model(batch_X)
            train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
            all_train_preds.extend(train_preds)
    train_preds = np.array(all_train_preds)
    train_f1 = f1_score(y_train, train_preds, average='macro')
    test_acc = f1_score(y_test, test_preds, average='micro')

    print(f"Final train F1: {train_f1:.4f}")
    print(f"Final validation F1: {val_f1:.4f}")
    print(f"Final test F1: {test_f1:.4f}")
    return model, train_f1, val_f1, test_f1, test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--c_values", type=float, nargs="+", default=[50.0, 100.0, 200.0])
    parser.add_argument("--pooling_types", type=str, nargs="+", default=["residual_mean"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.8])
    parser.add_argument("--n_trials", type=int, default=16)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--use_gpu_data", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Device: {device}")

    print("\n[1/3] Extracting representations...")
    all_reps = extract_all_representations(args.model, args.datasets, args.device, args.batch_size, args.pooling_types, args.val_ratio)

    print("\n[2/3] Training sparse probes...")
    best_probes = train_probes(all_reps["train"]["representations"], all_reps["train"]["labels"], all_reps["validation"]["representations"], all_reps["validation"]["labels"], all_reps["validation"]["dataset_ids"], all_reps["test"]["representations"], all_reps["test"]["labels"], all_reps["test"]["dataset_ids"], all_reps["train"]["num_layers"], args.c_values, args.pooling_types, device)

    os.makedirs("probes", exist_ok=True)
    probe_path = f"probes/{args.model}_general_probes.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({"best_probes": best_probes, "model": args.model, "dataset": "general"}, f)
    print(f"\nSaved probes to {probe_path}")

    print("\n[3/3] Training final MLP with neuron aggregation...")
    all_results = []
    for pooling_type in args.pooling_types:
        print(f"\nProcessing pooling_type={pooling_type}")
        layer_weights = get_layer_weights(best_probes, pooling_type, all_reps["train"]["num_layers"])
        for threshold in args.thresholds:
            print(f"\nTesting threshold={threshold}")
            selected_neurons_dict = {}
            for layer_idx in layer_weights.keys():
                key = f"layer{layer_idx}_{pooling_type}"
                if key in best_probes:
                    probe = best_probes[key]["probe"]
                    selected_indices = select_salient_neurons(probe, threshold)
                    selected_neurons_dict[key] = selected_indices

            X_train, selected_layers = aggregate_features(all_reps["train"]["representations"], pooling_type, selected_neurons_dict, layer_weights)
            X_val, _ = aggregate_features(all_reps["validation"]["representations"], pooling_type, selected_neurons_dict, layer_weights, selected_layers)
            X_test, _ = aggregate_features(all_reps["test"]["representations"], pooling_type, selected_neurons_dict, layer_weights, selected_layers)
            print(f"Feature dimension: {X_train.shape[1]}")

            best_params, cv_f1 = train_with_optuna(X_train, all_reps["train"]["labels"], all_reps["train"]["dataset_ids"], X_val, all_reps["validation"]["labels"], all_reps["validation"]["dataset_ids"], device, args.n_trials, args.n_jobs, n_folds=args.n_folds, use_cv=True, use_gpu_data=bool(args.use_gpu_data))
            print(f"\nTraining final model on full train+val for pooling={pooling_type}, threshold={threshold}...")
            X_full_train = np.vstack([X_train, X_val])
            y_full_train = np.concatenate([all_reps["train"]["labels"], all_reps["validation"]["labels"]])
            full_train_dataset_ids = np.concatenate([all_reps["train"]["dataset_ids"], all_reps["validation"]["dataset_ids"]])
            final_model, train_f1, final_val_f1, test_f1, test_acc = train_final_model(X_full_train, y_full_train, np.array([]), np.array([]), np.array([]), X_test, all_reps["test"]["labels"], all_reps["test"]["dataset_ids"], best_params, device, use_val=False, cv_f1=cv_f1, use_gpu_data=bool(args.use_gpu_data))
            result = {"pooling_type": pooling_type, "threshold": threshold, "layer_weight_strategy": "performance", "selected_layers": selected_layers, "num_features": int(X_train.shape[1]), "layer_weights": {str(k): float(v) for k, v in layer_weights.items()}, "selected_neurons_dict": {k: list(v) for k, v in selected_neurons_dict.items()}, "best_hyperparameters": best_params, "train_f1": float(train_f1), "val_f1": float(final_val_f1), "test_f1": float(test_f1), "test_acc": float(test_acc), "final_mlp": final_model}
            all_results.append(result)
            print(f"\nResult for pooling={pooling_type}, threshold={threshold}: val_f1={final_val_f1:.4f}, test_f1={test_f1:.4f}, test_acc={test_acc:.4f}")

    best_overall = max(all_results, key=lambda x: x['val_f1'])
    output_dir = f"probes/optuna/{args.model}_general"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best_overall, f)

    def convert_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_json(item) for item in obj]
        return obj

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump({"best_overall": convert_json({k: v for k, v in best_overall.items() if k != "final_mlp"}), "all_results": convert_json([{k: v for k, v in r.items() if k != "final_mlp"} for r in sorted(all_results, key=lambda x: x['val_f1'], reverse=True)])}, f, indent=2)

    print(f"\nSaved to: {output_dir}/best_model.pkl")

if __name__ == "__main__":
    main()
