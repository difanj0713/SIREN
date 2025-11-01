import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# pytorch-based linear probe with L1 regularization; no scikit-learn implementation here for gpu acceleration
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

    def train(self, X, y, X_val=None, y_val=None, quick_eval=False, random_seed=42):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        input_dim = X.shape[1]
        self.model = nn.Linear(input_dim, 2).to(self.device)
        
        nn.init.kaiming_normal_(self.model.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.model.bias)
        
        l1_lambda = 1.0 / self.C
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
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
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                if self.penalty == "l1":
                    l1_reg = torch.sum(torch.abs(self.model.weight)) + torch.sum(torch.abs(self.model.bias))
                    loss += l1_lambda * l1_reg
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if X_val is not None and y_val is not None:
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
                outputs = self.model(batch_X)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        
        return np.array(predictions)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy

    def get_feature_importance(self):
        if not self.trained:
            raise ValueError("Model not trained yet")
        with torch.no_grad():
            return torch.abs(self.model.weight[1] - self.model.weight[0]).cpu().numpy()

def extract_layer_features(representations, layer_idx, rep_type, pooling):
    key = f"{rep_type}_{pooling}"
    features = []
    for rep in representations:
        features.append(rep[layer_idx][key])
    return np.array(features)

def train_and_evaluate_probe(train_reps, train_labels, val_reps, val_labels, layer_idx, rep_type, pooling, C_values, device="cuda"):
    X_train = extract_layer_features(train_reps, layer_idx, rep_type, pooling)
    X_val = extract_layer_features(val_reps, layer_idx, rep_type, pooling)
    
    best_C = None
    best_val_acc = 0
    
    for C in C_values:
        probe = LinearProbe(C=C, device=device)
        probe.train(X_train, train_labels, X_val, val_labels, quick_eval=True)
        val_acc = probe.evaluate(X_val, val_labels)
        print(f"    C={C}: val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
    
    print(f"    Best C={best_C} with val_acc={best_val_acc:.4f}")
    
    final_probe = LinearProbe(C=best_C, device=device)
    final_probe.train(X_train, train_labels, X_val, val_labels, quick_eval=False)
    
    train_acc = final_probe.evaluate(X_train, train_labels)
    val_acc = final_probe.evaluate(X_val, val_labels)
    
    return final_probe, train_acc, val_acc, best_C
