#!/usr/bin/env python3
"""
Evaluate RAGRoute resource selection on FeB4RAG using nDCG@k and nP@k metrics.

Converts source_scores JSONL output from run_benchmark.py into a TREC run file,
then calls FW-eval-RS.py for evaluation.

Usage:
    python eval_resource_selection.py \
        --scores-file data/source_scores_FeB4RAG_ragroute.jsonl \
        --qrels-file /path/to/FeB4RAG/dataset/qrels/BEIR-QRELS-RS.txt \
        --trec-eval-path /tmp/trec_eval
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile


def scores_to_trec_run(scores_file, output_run_file, run_name="ragroute"):
    """Convert source_scores JSONL to TREC run format."""
    with open(scores_file, "r") as f_in, open(output_run_file, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            qid = record["question_id"]
            scores = record["source_scores"]
            # Sort sources by score descending
            ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            for rank, (engine, score) in enumerate(ranked):
                f_out.write(f"{qid} Q0 {engine} {rank} {score} {run_name}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate resource selection metrics")
    parser.add_argument("--scores-file", required=True, help="Path to source_scores JSONL from run_benchmark.py")
    parser.add_argument("--qrels-file", required=True, help="Path to BEIR-QRELS-RS.txt")
    parser.add_argument("--trec-eval-path", default="/tmp/trec_eval", help="Path to trec_eval directory")
    parser.add_argument("--output-run-file", default=None, help="Path to save TREC run file (default: auto)")
    parser.add_argument("--eval-script", default=None, help="Path to FW-eval-RS.py (default: auto-detect)")
    args = parser.parse_args()

    # Auto-detect eval script
    if args.eval_script is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "FeB4RAG", "dataset", "eval_script", "FW-eval-RS.py"),
            os.path.expanduser("~/FeB4RAG/dataset/eval_script/FW-eval-RS.py"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.eval_script = os.path.abspath(c)
                break
        if args.eval_script is None:
            print("Error: Could not find FW-eval-RS.py. Specify with --eval-script.")
            sys.exit(1)

    # Generate run file
    if args.output_run_file is None:
        base = os.path.splitext(args.scores_file)[0]
        args.output_run_file = base.replace("source_scores_", "run_") + ".txt"

    print(f"Converting {args.scores_file} -> {args.output_run_file}")
    run_name = os.path.basename(args.scores_file).replace("source_scores_", "").replace(".jsonl", "")
    scores_to_trec_run(args.scores_file, args.output_run_file, run_name=run_name)

    # Count queries
    with open(args.output_run_file) as f:
        qids = set(line.split()[0] for line in f)
    print(f"Generated run file with {len(qids)} queries")

    # Run evaluation
    print(f"\nRunning evaluation...")
    cmd = [sys.executable, args.eval_script, args.output_run_file, args.qrels_file, args.trec_eval_path]
    print(f"  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running eval script:\n{result.stderr}")
        sys.exit(1)

    print(result.stdout)


if __name__ == "__main__":
    main()
