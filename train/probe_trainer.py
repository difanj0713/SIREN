import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score


def compute_per_dataset_f1(y_true, y_pred, dataset_ids):
    unique_datasets = np.unique(dataset_ids)
    dataset_f1s = []
    for dataset_id in unique_datasets:
        mask = dataset_ids == dataset_id
        dataset_f1 = f1_score(y_true[mask], y_pred[mask], average='macro')
        dataset_f1s.append(dataset_f1)
    return np.mean(dataset_f1s)


class LinearProbe:
    def __init__(self, C=1.0, penalty="l1", max_iter=1000, device="cuda", batch_size=256):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.trained = False
        self.best_val_acc = 0
        self.best_model_state = None

    def train(self, X, y, X_val=None, y_val=None, val_dataset_ids=None, quick_eval=False, random_seed=42):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        input_dim = X.shape[1]
        self.model = nn.Linear(input_dim, 1).to(self.device)

        nn.init.kaiming_normal_(self.model.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.model.bias)

        l1_lambda = 1.0 / self.C
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        max_epochs = 100 if quick_eval else 500
        patience = 15 if quick_eval else 50
        patience_counter = 0
        
        self.best_val_acc = 0
        self.best_model_state = None
        
        self.trained = True
        self.model.train()
        
        for epoch in range(max_epochs):
            epoch_loss = 0
            num_batches = 0
            
            indices = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                if self.penalty == "l1":
                    l1_reg = torch.sum(torch.abs(self.model.weight)) + torch.sum(torch.abs(self.model.bias))
                    loss += l1_lambda * l1_reg
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if X_val is not None and y_val is not None:
                if val_dataset_ids is not None:
                    val_preds = self.predict(X_val)
                    val_acc = compute_per_dataset_f1(y_val, val_preds, val_dataset_ids)
                else:
                    val_acc = self.evaluate(X_val, y_val)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model not trained yet")
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
                outputs = self.model(batch_X).squeeze()
                probs = torch.sigmoid(outputs)
                batch_preds = (probs > 0.5).cpu().numpy().astype(int)
                predictions.extend(batch_preds)

        return np.array(predictions)

    def evaluate(self, X, y, dataset_ids=None, metric='accuracy'):
        predictions = self.predict(X)
        if metric == 'accuracy':
            return accuracy_score(y, predictions)
        elif metric == 'f1_macro':
            if dataset_ids is not None:
                return compute_per_dataset_f1(y, predictions, dataset_ids)
            else:
                return f1_score(y, predictions, average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_feature_importance(self):
        if not self.trained:
            raise ValueError("Model not trained yet")
        with torch.no_grad():
            return torch.abs(self.model.weight.squeeze()).cpu().numpy()

def extract_layer_features(representations, layer_idx, rep_type, pooling):
    key = f"{rep_type}_{pooling}"
    features = []
    for rep in representations:
        features.append(rep[layer_idx][key])
    return np.array(features)

def train_and_evaluate_probe(train_reps, train_labels, val_reps, val_labels, val_dataset_ids, layer_idx, rep_type, pooling, C_values, device="cuda", metric='accuracy'):
    X_train = extract_layer_features(train_reps, layer_idx, rep_type, pooling)
    X_val = extract_layer_features(val_reps, layer_idx, rep_type, pooling)

    best_C = None
    best_val_score = 0

    for C in C_values:
        probe = LinearProbe(C=C, device=device)
        probe.train(X_train, train_labels, X_val, val_labels, val_dataset_ids, quick_eval=True)
        val_score = probe.evaluate(X_val, val_labels, val_dataset_ids, metric=metric)
        print(f"    C={C}: val_{metric}={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_C = C

    print(f"    Best C={best_C} with val_{metric}={best_val_score:.4f}")

    final_probe = LinearProbe(C=best_C, device=device)
    final_probe.train(X_train, train_labels, X_val, val_labels, val_dataset_ids, quick_eval=False)

    train_score = final_probe.evaluate(X_train, train_labels, metric=metric)
    val_score = final_probe.evaluate(X_val, val_labels, val_dataset_ids, metric=metric)

    return final_probe, train_score, val_score, best_C
