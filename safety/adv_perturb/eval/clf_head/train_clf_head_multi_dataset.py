import sys
sys.path.append('../../../')
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from preprocess import preprocess_dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_classification_head(model, train_loader, val_loader, device, epochs=5, lr=1e-4, patience=1):
    for param in model.model.parameters():
        param.requires_grad = False

    for param in model.score.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.score.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.score.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.score.load_state_dict(best_model_state)

    return model, best_val_f1

def evaluate_clf_head(model, texts, labels, tokenizer, device, batch_size=256):
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    return {"f1": float(f1), "accuracy": float(acc)}

def train_and_eval_dataset(dataset_name, model, tokenizer, device, epochs=5, batch_size=256, patience=1, save_dir="clf_heads"):
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name}")
    print(f"{'='*80}")

    dataset_dict = preprocess_dataset(dataset_name)
    train_data = dataset_dict["train"]
    val_data = dataset_dict["validation"]
    test_data = dataset_dict["test"]

    train_texts = [sample["text"] for sample in train_data]
    train_labels = [sample["label"] for sample in train_data]
    val_texts = [sample["text"] for sample in val_data]
    val_labels = [sample["label"] for sample in val_data]
    test_texts = [sample["text"] for sample in test_data]
    test_labels = [sample["label"] for sample in test_data]

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model, best_val_f1 = train_classification_head(model, train_loader, val_loader, device, epochs, patience=patience)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_clf_head.pt")
    torch.save(model.score.state_dict(), save_path)
    print(f"Saved clf head to {save_path}")

    test_results = evaluate_clf_head(model, test_texts, test_labels, tokenizer, device, batch_size)
    print(f"Test F1: {test_results['f1']:.4f}, Test Acc: {test_results['accuracy']:.4f}")

    return {
        "dataset": dataset_name,
        "val_f1": float(best_val_f1),
        "test_f1": test_results["f1"],
        "test_acc": test_results["accuracy"]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='clf_heads')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from config import MODEL_CONFIGS
    model_path = MODEL_CONFIGS[args.model]["model_path"]

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    save_dir = f"{args.save_dir}/{args.model}"
    all_results = []

    for dataset_name in args.datasets:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            trust_remote_code=True,
            pad_token_id=tokenizer.pad_token_id
        ).to(device)

        result = train_and_eval_dataset(
            dataset_name, model, tokenizer, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            save_dir=save_dir
        )
        all_results.append(result)

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} | {'Val F1':>7} | {'Test F1':>7} | {'Test Acc':>7}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['dataset']:<20} | {result['val_f1']:7.4f} | {result['test_f1']:7.4f} | {result['test_acc']:7.4f}")

    os.makedirs("results", exist_ok=True)
    output_path = f"results/{args.model}_clf_head_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
