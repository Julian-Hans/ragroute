import os
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir, output_dir, include_source_id=True, include_centroid=True,
              train_qids=None):
    """Load pre-computed features and split into train/val(/test).

    If train_qids is provided, only those queries are used for training
    (with 20% held out for validation). No test set is created — evaluation
    is handled externally.

    Otherwise falls back to loading/creating a random train/val/test split
    via split.json (legacy CLI mode).
    """
    embeddings_dir = os.path.join(data_dir, "embeddings")

    with open(os.path.join(embeddings_dir, "routing_grouped_by_query.pkl"), "rb") as f:
        query_to_data = pickle.load(f)

    with open(os.path.join(embeddings_dir, "encoder_dims.json"), "r") as f:
        encoder_dims = json.load(f)
    max_dim = max(encoder_dims.values())

    with open(os.path.join(embeddings_dir, "source_id_map.json"), "r") as f:
        source_to_id = json.load(f)
    num_sources = len(source_to_id)

    def flatten(qids):
        return [ex for qid in qids for ex in query_to_data[qid]]

    def unpack(data):
        X_all, y_all = [], []
        for x, y in data:
            q_vec = x[:max_dim]
            c_vec = x[max_dim:2*max_dim]
            s_vec = x[2*max_dim:]
            parts = [q_vec]
            if include_centroid:
                parts.append(c_vec)
            if include_source_id:
                parts.append(s_vec)
            X_all.append(np.concatenate(parts))
            y_all.append(y)
        return np.array(X_all), np.array(y_all)

    if train_qids is not None:
        available = set(query_to_data.keys())
        train_qids = [q for q in train_qids if q in available]
        train_q, val_q = train_test_split(train_qids, test_size=0.2, random_state=42)
        print(f"Custom split: {len(train_q)} train, {len(val_q)} val")
        X_train, y_train = unpack(flatten(train_q))
        X_val, y_val = unpack(flatten(val_q))
        return (X_train, y_train), (X_val, y_val), None, num_sources if include_source_id else 0

    # Legacy: random split with test set
    split_path = os.path.join(output_dir, "split.json")
    query_ids = list(query_to_data.keys())

    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split_data = json.load(f)
        train_q = split_data["train"]
        val_q = split_data["val"]
        test_q = split_data["test"]
        print(f"Loaded existing split from {split_path}")
    else:
        train_q, rest_q = train_test_split(query_ids, test_size=0.7, random_state=42)
        val_q, test_q = train_test_split(rest_q, test_size=6/7, random_state=42)
        split_data = {"train": train_q, "val": val_q, "test": test_q}
        os.makedirs(output_dir, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved new split to {split_path}")

    X_train, y_train = unpack(flatten(train_q))
    X_val, y_val = unpack(flatten(val_q))
    X_test, y_test = unpack(flatten(test_q))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_sources if include_source_id else 0


class RoutingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class CorpusRouter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x).squeeze()


def evaluate_with_metrics(model, loader, device, threshold=0.5):
    print("THRESHOLD ", threshold)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    TP = FP = FN = TN = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

    acc = accuracy_score(all_labels, all_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0

    num_total = len(all_labels)
    num_true_positives = sum(all_labels)
    num_pred_positives = sum(all_preds)
    reduction_pred = 1 - (num_pred_positives / num_total)
    reduction_true = 1 - (num_true_positives / num_total)

    print(f"Total test instances: {num_total}")
    print(f"Ground-truth positives: {int(num_true_positives)}, reduction {reduction_true*100}")
    print(f"Predicted positives:   {int(num_pred_positives)}, reduction {reduction_pred*100}")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    print(f"Acc: {acc:.2%} | Prec: {prec:.2%} | Recall: {recall:.2%} | F1: {f1:.2%} | AUC: {auc:.2f}")
    return acc, prec, recall, f1, auc


def train_router(data_dir, output_dir, include_source_id=True, include_centroid=True,
                 num_epochs=150, train_qids=None):
    """Train a CorpusRouter model and save the best checkpoint.

    Args:
        data_dir: Directory containing embeddings/ from prepare_feb4rag_data.py
        output_dir: Directory to save trained model and split.json
        include_source_id: Include one-hot source ID in features
        include_centroid: Include centroid embedding in features
        num_epochs: Number of training epochs
        train_qids: Optional list of query IDs for training. If provided,
            20% is held out for validation and no test set is created
            (evaluation is handled externally). If None, uses a random
            train/val/test split via split.json.

    Returns:
        Tuple of (model, best_model_path, test_metrics) where test_metrics is
        (acc, prec, recall, f1, auc) or None if no test set.
    """
    set_seed()

    os.makedirs(output_dir, exist_ok=True)
    (X_train, y_train), (X_val, y_val), test_data, _ = load_data(
        data_dir, output_dir, include_source_id, include_centroid,
        train_qids=train_qids)

    train_loader = DataLoader(RoutingDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader   = DataLoader(RoutingDataset(X_val, y_val), batch_size=128)
    test_loader  = DataLoader(RoutingDataset(*test_data), batch_size=128) if test_data else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorpusRouter(input_dim=X_train.shape[1]).to(device)

    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-3,
        max_lr=5e-3,
        step_size_up=10,
        mode="triangular2",
        cycle_momentum=False,
    )
    scheduler_fixed = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

    print("\nStarting training...")
    best_acc = 0
    best_model_path = os.path.join(output_dir, "router_best_model.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch < 115:
                scheduler.step()
            else:
                scheduler_fixed.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch}: Train Loss = {avg_loss:.4f}")
        print("Validation Set Metrics:")
        acc, _, _, f1, _ = evaluate_with_metrics(model, val_loader, device)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1 = {f1:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    print("\nLoaded best model from disk.")

    test_metrics = None
    if test_loader is not None:
        print("\nFinal Test Set Evaluation:")
        test_metrics = evaluate_with_metrics(model, test_loader, device)

    return model, best_model_path, test_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/julian/ragroute/data/feb4rag_embeddings",
                        help="Directory containing embeddings/ from prepare_feb4rag_data.py")
    parser.add_argument("--output-dir", default="/home/julian/ragroute/data/feb4rag_router",
                        help="Directory to save trained model and split.json")
    args = parser.parse_args()

    train_router(args.data_dir, args.output_dir)
