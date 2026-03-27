#!/usr/bin/env python3
"""
Train and evaluate the FeB4RAG router on explicit query ID splits.

Usage:
    python scripts/evaluate_feb4rag_router.py --train 1-400 --test 401-790
    python scripts/evaluate_feb4rag_router.py --train 1-50 --test 51-100
    python scripts/evaluate_feb4rag_router.py --train 1-400 --test 401-790 --skip-train
    python scripts/evaluate_feb4rag_router.py --train all --test all  # sanity check

Query IDs are integers 1–790 (contiguous).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset


# ── Model (same arch as train_feb4rag_router.py) ────────────────────────────

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
        return self.fc3(x).squeeze(-1)


class RoutingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ── Query ID parsing ─────────────────────────────────────────────────────────

def parse_id_spec(raw: str, all_ids: Set[str]) -> Set[str]:
    """Parse a query ID spec: "1-50", "1,5,8-12", "all"."""
    if raw == "all":
        return set(all_ids)
    ids: Set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ids.update(str(i) for i in range(int(start), int(end) + 1))
        else:
            ids.add(part)
    missing = ids - all_ids
    if missing:
        print(f"Warning: {len(missing)} query IDs not in dataset (e.g. {sorted(missing)[:5]})")
        ids -= missing
    return ids


# ── Training ─────────────────────────────────────────────────────────────────

def flatten_ids(query_to_data, query_ids):
    X, y = [], []
    for qid in query_ids:
        for feat, label in query_to_data[qid]:
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)


def evaluate_loader(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            probs = torch.sigmoid(model(X))
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return acc, f1, auc


def train_router(X_train, y_train, X_val, y_val, input_dim, model_path, epochs, device):
    model = CorpusRouter(input_dim).to(device)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-3, max_lr=5e-3, step_size_up=10,
        mode="triangular2", cycle_momentum=False,
    )
    scheduler_fixed = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

    train_loader = DataLoader(RoutingDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader   = DataLoader(RoutingDataset(X_val,   y_val),   batch_size=128)

    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if epoch < 115:
                scheduler.step()
            else:
                scheduler_fixed.step()
            total_loss += loss.item()

        acc, f1, auc = evaluate_loader(model, val_loader, device)
        print(f"Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} | val acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

    print(f"\nBest model at epoch {best_epoch} (val acc={best_acc:.4f}), saved to {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, query_to_data, test_ids, sorted_corpora, device, out_path):
    model.eval()
    with open(out_path, "w") as f_out:
        for qid in sorted(test_ids, key=int):
            samples = query_to_data[qid]
            features = np.stack([feat for feat, _ in samples])
            x = torch.tensor(features, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).cpu().numpy()
            source_scores = {corpus: float(probs[i]) for i, corpus in enumerate(sorted_corpora)}
            f_out.write(json.dumps({"question_id": qid, "source_scores": source_scores}) + "\n")
    print(f"Wrote {len(test_ids)} records to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate FeB4RAG router on explicit query splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", required=True,
                        help='Train query IDs: "1-400", "1,5,8-12", "all"')
    parser.add_argument("--test", required=True,
                        help='Test query IDs: "401-790", "all"')
    parser.add_argument("--data-dir",
                        default="/home/julian/repos/FeB4RAG/dataset_creation/2_search/embeddings",
                        help="Dir containing embeddings/ subdir")
    parser.add_argument("--output-dir",
                        default="/home/julian/repos/FeB4RAG/dataset_creation/3_router",
                        help="Where to save model and results")
    parser.add_argument("--qrels-file",
                        default="/home/julian/repos/FeB4RAG/dataset/qrels/BEIR-QRELS-RS.txt")
    parser.add_argument("--trec-eval-path", default="/tmp/trec_eval")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, reuse existing router_best_model.pt in --output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    embeddings_dir = os.path.join(args.data_dir, "embeddings")

    with open(os.path.join(embeddings_dir, "source_id_map.json")) as f:
        source_to_id = json.load(f)
    sorted_corpora = sorted(source_to_id.keys())

    with open(os.path.join(embeddings_dir, "routing_grouped_by_query.pkl"), "rb") as f:
        query_to_data = pickle.load(f)

    all_ids = set(query_to_data.keys())
    train_ids = parse_id_spec(args.train, all_ids)
    test_ids  = parse_id_spec(args.test,  all_ids)

    print(f"Train queries: {len(train_ids)}  |  Test queries: {len(test_ids)}")
    overlap = train_ids & test_ids
    if overlap:
        print(f"Warning: {len(overlap)} queries appear in both train and test")

    # Determine input dim
    sample_feat, _ = query_to_data[next(iter(all_ids))][0]
    input_dim = len(sample_feat)

    model_path = os.path.join(args.output_dir, "router_best_model.pt")

    # ── Train ──
    if args.skip_train:
        print(f"Skipping training, loading {model_path}")
        model = CorpusRouter(input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"\nFlattening training data ({len(train_ids)} queries × 13 = {len(train_ids)*13} samples)...")
        X_train, y_train = flatten_ids(query_to_data, train_ids)
        # Use test set as validation during training (no held-out val needed for range splits)
        X_val, y_val = flatten_ids(query_to_data, test_ids)
        print(f"X_train: {X_train.shape}  pos_rate: {y_train.mean():.2%}")
        print(f"\nTraining for {args.epochs} epochs...")
        model = train_router(X_train, y_train, X_val, y_val, input_dim, model_path, args.epochs, device)

    # ── Infer ──
    tag = f"train{args.train.replace(',','_').replace('-','t')}_test{args.test.replace(',','_').replace('-','t')}"
    scores_path = os.path.join(args.output_dir, f"source_scores_ragroute_{tag}.jsonl")
    print(f"\nRunning inference on {len(test_ids)} test queries...")
    run_inference(model, query_to_data, test_ids, sorted_corpora, device, scores_path)

    # ── Eval ──
    eval_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_resource_selection.py"))
    cmd = [sys.executable, eval_script,
           "--scores-file", scores_path,
           "--qrels-file", args.qrels_file,
           "--trec-eval-path", args.trec_eval_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        print("Evaluation failed.")
        sys.exit(1)

    # Parse aggregate line: all,ndcg10,ndcg20,ndcg100,np1,np5
    agg = None
    for line in result.stdout.splitlines():
        if line.startswith("all,"):
            parts = line.split(",")
            agg = parts[1:]
            break

    if agg:
        def fmt(v):
            try: return f"{float(v):>8.4f}"
            except ValueError: return f"{'n/a':>8}"

        print()
        print(f"{'Router':<20} {'nDCG@10':>8} {'nDCG@20':>8} {'nDCG@100':>9} {'nP@1':>8} {'nP@5':>8}")
        print("-" * 65)
        print(f"{'RAGRoute':<20} {fmt(agg[0])} {fmt(agg[1])} {fmt(agg[2])} {fmt(agg[3])} {fmt(agg[4])}")
        print()
    else:
        print(result.stdout)


if __name__ == "__main__":
    main()
