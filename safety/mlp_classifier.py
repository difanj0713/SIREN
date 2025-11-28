import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

class MLPClassifier:
    def __init__(self, input_dim, hidden_dim, num_classes=2, dropout=0.1, device="cuda"):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)

        self.trained = False
        self.best_val_f1 = 0
        self.best_model_state = None

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001, patience=15):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        self.best_val_f1 = 0
        self.best_model_state = None
        patience_counter = 0

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            val_f1 = self.evaluate(X_val, y_val, metric='f1_macro')

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})

        self.trained = True

    def predict(self, X):
        self.model.eval()
        predictions = []

        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                outputs = self.model(batch_X)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_preds)

        return np.array(predictions)

    def evaluate(self, X, y, metric='f1_macro'):
        predictions = self.predict(X)
        if metric == 'f1_macro':
            return f1_score(y, predictions, average='macro')
        elif metric == 'accuracy':
            return accuracy_score(y, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric}")
