#!/usr/bin/env python3
"""
Prepare training data for the FeB4RAG router model.

Generates:
  - embeddings/routing_grouped_by_query.pkl  (training data)
  - embeddings/encoder_dims.json             (model name -> embedding dim)
  - embeddings/source_id_map.json            (source name -> int ID)
  - {corpus}_{model}_stats.json              (centroids for inference)

Usage:
  python prepare_feb4rag_data.py \
      --output-dir ../../data/feb4rag_embeddings \
      --corpus-dir ~/FeB4RAG/dataset/original_dataset \
      --queries-path ~/FeB4RAG/dataset/queries/requests.jsonl \
      --qrels-path ~/FeB4RAG/dataset/qrels/BEIR-QRELS-RS.txt \
      --model-dir ~/ragroute/data/feb4rag_models \
      --resume
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path so we can import ragroute modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ragroute.models.feb4rag.model_zoo import BeirModels, CustomModel

# ── Constants (from ragroute/ragroute/config.py) ────────────────────────────

CORPUS_MODEL_MAP = {
    "msmarco": ("e5-large", "custom"),
    "trec-covid": ("SGPT-5.8B-weightedmean-msmarco-specb-bitfit", "custom"),
    "nfcorpus": ("UAE-Large-V1", "custom"),
    "scidocs": ("all-mpnet-base-v2", "beir"),
    "nq": ("multilingual-e5-large", "custom"),
    "hotpotqa": ("ember-v1", "beir"),
    "fiqa": ("all-mpnet-base-v2", "beir"),
    "arguana": ("UAE-Large-V1", "custom"),
    "webis-touche2020": ("e5-base", "custom"),
    "dbpedia-entity": ("UAE-Large-V1", "custom"),
    "fever": ("UAE-Large-V1", "custom"),
    "climate-fever": ("UAE-Large-V1", "custom"),
    "scifact": ("gte-base", "beir"),
}

SOURCE_TO_ID = {
    "arguana": 0, "climate-fever": 1, "dbpedia-entity": 2, "fever": 3,
    "fiqa": 4, "hotpotqa": 5, "msmarco": 6, "nfcorpus": 7, "nq": 8,
    "scidocs": 9, "scifact": 10, "trec-covid": 11, "webis-touche2020": 12,
}

MAX_DIM = 4096
NUM_SOURCES = 13

# Batch sizes tuned for H100 80GB
CORPUS_BATCH_SIZES = {
    "SGPT-5.8B-weightedmean-msmarco-specb-bitfit": 256,
    "UAE-Large-V1": 256,
    "e5-large": 256,
    "e5-base": 512,
    "multilingual-e5-large": 128,
    "all-mpnet-base-v2": 512,
    "ember-v1": 512,
    "gte-base": 512,
}

QUERY_BATCH_SIZES = {
    "SGPT-5.8B-weightedmean-msmarco-specb-bitfit": 512,
    "UAE-Large-V1": 512,
    "e5-large": 512,
    "e5-base": 1024,
    "multilingual-e5-large": 256,
    "all-mpnet-base-v2": 1024,
    "ember-v1": 1024,
    "gte-base": 1024,
}


def load_queries(path):
    """Load queries from requests.jsonl."""
    queries = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            queries[str(obj["_id"])] = obj["text"]
    print(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(path, valid_sources):
    """Load resource-selection qrels. Returns {qid: {source: score}}."""
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, source, score = parts
            if source in valid_sources:
                qrels[qid][source] = int(score)
    print(f"Loaded qrels for {len(qrels)} queries")
    return dict(qrels)


def read_corpus_sample(corpus_path, sample_size):
    """Reservoir-sample sample_size docs from corpus JSONL in one pass."""
    import random
    reservoir = []
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            item = {"title": doc.get("title", ""), "text": doc.get("text", "")}
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = item
    return reservoir


def read_corpus_streaming(corpus_path, chunk_size):
    """Yield chunks of corpus documents from JSONL file."""
    chunk = []
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            chunk.append({"title": doc.get("title", ""), "text": doc.get("text", "")})
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def count_corpus_lines(corpus_path):
    """Fast line count for progress bar."""
    count = 0
    with open(corpus_path, "rb") as f:
        for _ in f:
            count += 1
    return count


def compute_centroid(model, corpus_path, batch_size, chunk_size, sample_size=None):
    """Compute centroid (mean embedding) of a corpus via streaming or random sample."""
    total_lines = count_corpus_lines(corpus_path)
    corpus_name = os.path.basename(os.path.dirname(os.path.dirname(corpus_path)))

    if sample_size is not None and sample_size < total_lines:
        print(f"  Corpus: {corpus_name} ({total_lines:,} docs, sampling {sample_size:,})")
        docs = read_corpus_sample(corpus_path, sample_size)
        embeddings = model.encode_corpus(docs, batch_size=batch_size, convert_to_tensor=False,
                                         show_progress_bar=True)
        centroid = embeddings.mean(axis=0).astype(np.float32)
        return centroid, len(docs)

    total_chunks = (total_lines + chunk_size - 1) // chunk_size
    print(f"  Corpus: {corpus_name} ({total_lines:,} docs, {total_chunks} chunks)")

    running_sum = None
    total_count = 0
    for chunk in tqdm(read_corpus_streaming(corpus_path, chunk_size),
                      total=total_chunks, desc=f"  Encoding {corpus_name}"):
        embeddings = model.encode_corpus(chunk, batch_size=batch_size, convert_to_tensor=False)
        if running_sum is None:
            running_sum = embeddings.sum(axis=0).astype(np.float64)
        else:
            running_sum += embeddings.sum(axis=0).astype(np.float64)
        total_count += len(embeddings)

    centroid = (running_sum / total_count).astype(np.float32)
    return centroid, total_count


def load_model(model_name, model_type, model_dir):
    """Load an embedding model."""
    if model_type == "beir":
        loader = BeirModels(model_dir, specific_model=model_name)
        # Download if not present
        model_name_mapped = None
        for name in loader.model_name_or_path:
            if model_name in name:
                model_name_mapped = name
                break
        local_path = os.path.join(model_dir, model_name_mapped.replace("/", "_"))
        if not os.path.exists(local_path):
            print(f"  Downloading {model_name_mapped} to {model_dir}...")
            loader.download_models()
        model = loader.load_model(model_name, cuda=torch.cuda.is_available())
    else:
        loader = CustomModel(model_dir=model_dir, specific_model=model_name)
        model = loader.load_model(model_name, cuda=torch.cuda.is_available())
    return model


def free_model(model):
    """Free GPU memory."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare FeB4RAG router training data")
    parser.add_argument("--output-dir", default="/home/julian/ragroute/data/feb4rag_embeddings",
                        help="Output directory for all generated files")
    parser.add_argument("--corpus-dir", default="/home/julian/FeB4RAG/dataset/original_dataset",
                        help="Directory containing corpus subdirs")
    parser.add_argument("--queries-path", default="/home/julian/FeB4RAG/dataset/queries/requests.jsonl",
                        help="Path to requests.jsonl")
    parser.add_argument("--qrels-path", default="/home/julian/FeB4RAG/dataset/qrels/BEIR-QRELS-RS.txt",
                        help="Path to BEIR-QRELS-RS.txt")
    parser.add_argument("--model-dir", default="/home/julian/ragroute/data/feb4rag_models",
                        help="Directory for model downloads/cache")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-computed embeddings and centroids")
    parser.add_argument("--chunk-size", type=int, default=50000,
                        help="Documents per encoding chunk for centroid computation")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Randomly sample this many docs per corpus for centroid (None = all docs)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directories
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Load queries and qrels
    queries = load_queries(args.queries_path)
    sorted_query_ids = sorted(queries.keys(), key=int)
    qrels = load_qrels(args.qrels_path, set(SOURCE_TO_ID.keys()))

    # Group corpora by model
    model_to_corpora = defaultdict(list)
    for corpus, (model_name, model_type) in CORPUS_MODEL_MAP.items():
        model_to_corpora[(model_name, model_type)].append(corpus)

    # Storage for assembly step
    encoder_dims = {}
    query_embeddings_all = {}  # model_name -> {qid: np.array}
    centroids_all = {}         # corpus_name -> np.array

    # Process each model sequentially
    for (model_name, model_type), corpora in model_to_corpora.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_type})")
        print(f"Corpora: {corpora}")
        print(f"{'='*60}")

        # ── Check if all work for this model is already cached ──
        query_cache_path = os.path.join(cache_dir, f"{model_name}_query_embeddings.npy")
        query_ids_cache_path = os.path.join(cache_dir, f"{model_name}_query_ids.json")
        all_stats_exist = all(
            os.path.exists(os.path.join(args.output_dir, f"{c}_{model_name}_stats.json"))
            for c in corpora
        )
        query_cache_exists = os.path.exists(query_cache_path)

        if args.resume and query_cache_exists and all_stats_exist:
            print(f"  [RESUME] All cached — loading from disk")
            embs = np.load(query_cache_path)
            with open(query_ids_cache_path) as f:
                cached_qids = json.load(f)
            query_embeddings_all[model_name] = {qid: embs[i] for i, qid in enumerate(cached_qids)}
            encoder_dims[model_name] = embs.shape[1]
            for corpus in corpora:
                stats_path = os.path.join(args.output_dir, f"{corpus}_{model_name}_stats.json")
                with open(stats_path) as f:
                    stats = json.load(f)
                centroids_all[corpus] = np.array(stats["centroid"], dtype=np.float32)
            continue

        # ── Load model ──
        t0 = time.time()
        print(f"  Loading model...")
        model = load_model(model_name, model_type, args.model_dir)
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        # ── Encode queries ──
        if args.resume and query_cache_exists:
            print(f"  [RESUME] Loading cached query embeddings")
            embs = np.load(query_cache_path)
            with open(query_ids_cache_path) as f:
                cached_qids = json.load(f)
            query_embeddings_all[model_name] = {qid: embs[i] for i, qid in enumerate(cached_qids)}
            encoder_dims[model_name] = embs.shape[1]
        else:
            print(f"  Encoding {len(sorted_query_ids)} queries...")
            query_texts = [queries[qid] for qid in sorted_query_ids]
            q_batch = QUERY_BATCH_SIZES.get(model_name, 128)
            t0 = time.time()
            query_embs = model.encode_queries(query_texts, batch_size=q_batch, convert_to_tensor=False)
            print(f"  Queries encoded in {time.time()-t0:.1f}s, shape: {query_embs.shape}")

            encoder_dims[model_name] = query_embs.shape[1]
            query_embeddings_all[model_name] = {
                qid: query_embs[i] for i, qid in enumerate(sorted_query_ids)
            }

            # Cache to disk
            np.save(query_cache_path, query_embs)
            with open(query_ids_cache_path, "w") as f:
                json.dump(sorted_query_ids, f)
            print(f"  Saved query embedding cache")

        # ── Compute centroids per corpus ──
        c_batch = CORPUS_BATCH_SIZES.get(model_name, 128)
        for corpus in corpora:
            stats_path = os.path.join(args.output_dir, f"{corpus}_{model_name}_stats.json")

            if args.resume and os.path.exists(stats_path):
                print(f"  [RESUME] Loading cached centroid for {corpus}")
                with open(stats_path) as f:
                    stats = json.load(f)
                centroids_all[corpus] = np.array(stats["centroid"], dtype=np.float32)
                continue

            corpus_path = os.path.join(args.corpus_dir, corpus, corpus, "corpus.jsonl")
            if not os.path.exists(corpus_path):
                print(f"  WARNING: Corpus not found: {corpus_path}")
                print(f"  Creating zero centroid for {corpus}")
                centroids_all[corpus] = np.zeros(encoder_dims[model_name], dtype=np.float32)
                continue

            t0 = time.time()
            centroid, num_docs = compute_centroid(model, corpus_path, c_batch, args.chunk_size,
                                                  sample_size=args.sample_size)
            elapsed = time.time() - t0
            print(f"  Centroid computed in {elapsed:.1f}s ({num_docs:,} docs, dim={len(centroid)})")

            centroids_all[corpus] = centroid
            stats = {"centroid": centroid.tolist(), "num_documents": num_docs,
                     "sample_size": args.sample_size}
            with open(stats_path, "w") as f:
                json.dump(stats, f)
            print(f"  Saved {stats_path}")

        # ── Free model ──
        free_model(model)

    # ── Assemble training pickle ──
    print(f"\n{'='*60}")
    print("Assembling training data...")
    print(f"{'='*60}")

    query_to_data = {}
    label_stats = {"positive": 0, "negative": 0}

    for qid in tqdm(sorted_query_ids, desc="Building features"):
        samples = []
        for corpus in sorted(SOURCE_TO_ID.keys()):
            model_name = CORPUS_MODEL_MAP[corpus][0]

            # Query embedding, padded
            q_emb = query_embeddings_all[model_name][qid]
            padded_q = np.pad(q_emb, (0, MAX_DIM - len(q_emb)))

            # Centroid, padded
            centroid = centroids_all[corpus]
            padded_c = np.pad(centroid, (0, MAX_DIM - len(centroid)))

            # One-hot source ID
            sid = SOURCE_TO_ID[corpus]
            one_hot = np.eye(NUM_SOURCES, dtype=np.float32)[sid]

            features = np.concatenate([padded_q, padded_c, one_hot])

            # Label from qrels
            label = 1 if qrels.get(qid, {}).get(corpus, 0) > 0 else 0
            if label:
                label_stats["positive"] += 1
            else:
                label_stats["negative"] += 1

            samples.append((features, label))

        query_to_data[qid] = samples

    # Verify dimensions
    sample_features = query_to_data[sorted_query_ids[0]][0][0]
    print(f"Feature vector dimension: {len(sample_features)} (expected {MAX_DIM*2 + NUM_SOURCES} = {MAX_DIM*2 + NUM_SOURCES})")
    assert len(sample_features) == MAX_DIM * 2 + NUM_SOURCES, "Feature dimension mismatch!"
    print(f"Total samples: {label_stats['positive'] + label_stats['negative']}")
    print(f"  Positive: {label_stats['positive']} ({label_stats['positive']/(label_stats['positive']+label_stats['negative'])*100:.1f}%)")
    print(f"  Negative: {label_stats['negative']} ({label_stats['negative']/(label_stats['positive']+label_stats['negative'])*100:.1f}%)")

    # Save pickle
    pkl_path = os.path.join(embeddings_dir, "routing_grouped_by_query.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(query_to_data, f)
    print(f"Saved {pkl_path}")

    # Save encoder dims
    dims_path = os.path.join(embeddings_dir, "encoder_dims.json")
    with open(dims_path, "w") as f:
        json.dump(encoder_dims, f, indent=2)
    print(f"Saved {dims_path}")

    # Save source ID map
    sid_path = os.path.join(embeddings_dir, "source_id_map.json")
    with open(sid_path, "w") as f:
        json.dump(SOURCE_TO_ID, f, indent=2)
    print(f"Saved {sid_path}")

    print(f"\nDone! All files saved to {args.output_dir}")
    print(f"Encoder dimensions: {encoder_dims}")


if __name__ == "__main__":
    main()
