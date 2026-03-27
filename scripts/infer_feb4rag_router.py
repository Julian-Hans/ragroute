#!/usr/bin/env python3
"""
Run trained FeB4RAG router on the test split and write source_scores.jsonl.

Output can be fed directly to eval_resource_selection.py.

Usage:
    python scripts/infer_feb4rag_router.py \
        --data-dir /home/julian/repos/FeB4RAG/dataset_creation/2_search/embeddings \
        --model-dir /home/julian/repos/FeB4RAG/dataset_creation/3_router \
        --output-dir /home/julian/repos/FeB4RAG/dataset_creation/3_router
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def parse_args():
    parser = argparse.ArgumentParser(description="FeB4RAG router inference")
    parser.add_argument("--data-dir",
                        default="/home/julian/repos/FeB4RAG/dataset_creation/2_search/embeddings",
                        help="Dir containing embeddings/ subdir (output of prepare_feb4rag_data.py)")
    parser.add_argument("--model-dir",
                        default="/home/julian/repos/FeB4RAG/dataset_creation/3_router",
                        help="Dir containing router_best_model.pt and split.json")
    parser.add_argument("--output-dir",
                        default="/home/julian/repos/FeB4RAG/dataset_creation/3_router",
                        help="Where to write source_scores_ragroute_feb4rag.jsonl")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test",
                        help="Which split to evaluate (default: test)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_dir = os.path.join(args.data_dir, "embeddings")

    # Load supporting files
    with open(os.path.join(embeddings_dir, "source_id_map.json")) as f:
        source_to_id = json.load(f)
    id_to_source = {v: k for k, v in source_to_id.items()}
    num_sources = len(source_to_id)

    with open(os.path.join(args.model_dir, "split.json")) as f:
        split = json.load(f)
    query_ids = split[args.split]
    print(f"Evaluating {len(query_ids)} queries from '{args.split}' split")

    with open(os.path.join(embeddings_dir, "routing_grouped_by_query.pkl"), "rb") as f:
        query_to_data = pickle.load(f)

    # Determine input dim from first sample
    sample_features, _ = query_to_data[query_ids[0]][0]
    input_dim = len(sample_features)
    print(f"Input dim: {input_dim}")

    # Load model
    model = CorpusRouter(input_dim=input_dim).to(device)
    model_path = os.path.join(args.model_dir, "router_best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Inference
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "source_scores_ragroute_feb4rag.jsonl")

    with open(out_path, "w") as f_out:
        for qid in query_ids:
            samples = query_to_data[qid]  # list of (feature_vec, label), one per corpus
            features = np.stack([feat for feat, _ in samples])  # (13, input_dim)
            x = torch.tensor(features, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(x)           # (13,)
                probs = torch.sigmoid(logits).cpu().numpy()  # (13,)

            # Map back to corpus names using the order in samples
            # samples are ordered by sorted(SOURCE_TO_ID.keys()), same as training
            sorted_corpora = sorted(source_to_id.keys())
            source_scores = {corpus: float(probs[i]) for i, corpus in enumerate(sorted_corpora)}

            f_out.write(json.dumps({"question_id": qid, "source_scores": source_scores}) + "\n")

    print(f"Wrote {len(query_ids)} records to {out_path}")
    print(f"\nNow run:")
    print(f"  python eval_resource_selection.py \\")
    print(f"    --scores-file {out_path} \\")
    print(f"    --qrels-file /home/julian/repos/FeB4RAG/dataset/qrels/BEIR-QRELS-RS.txt")


if __name__ == "__main__":
    main()
