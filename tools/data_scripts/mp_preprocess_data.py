#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-node parallel data preprocessing script for Megatron-LM.

This script distributes .jsonl files across multiple nodes/ranks for parallel
tokenization. Each rank processes a subset of files and calls tools/preprocess_data.py
to perform the actual tokenization.

Environment variables used for distributed setup:
    RANK: Current node rank (0-indexed)
    RC_WORLD_SIZE: Total number of nodes

Usage:
    # Single node (default)
    python tools/data_scripts/pai_preprocess_data.py \
        --data-root /path/to/data \
        --category category1 category2 \
        --tokenizer-model /path/to/tokenizer \
        --output-path /path/to/output \
        --workers 24

    # Multi-node (via PAI or similar launcher)
    # Each node runs the same command with different RANK env var
    RANK=0 RC_WORLD_SIZE=4 python tools/data_scripts/pai_preprocess_data.py ...
"""

import os
import sys
import glob
import argparse
import subprocess
import random
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-node parallel data preprocessing for Megatron-LM"
    )

    # Input data options
    input_group = parser.add_argument_group("Input data")
    input_group.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing data categories (used with --category)"
    )
    input_group.add_argument(
        "--category",
        nargs="+",
        type=str,
        default=[],
        help="Category subdirectories under --data-root to process"
    )
    input_group.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob pattern to match input files directly (alternative to --data-root + --category)"
    )
    input_group.add_argument(
        "--json-keys",
        nargs="+",
        default=["text"],
        help="Space-separated list of keys to extract from JSON (default: text)"
    )

    # Tokenizer options
    tokenizer_group = parser.add_argument_group("Tokenizer")
    tokenizer_group.add_argument(
        "--tokenizer-model",
        "--tokenizer-path",
        type=str,
        dest="tokenizer_model",
        required=True,
        help="Path to the HuggingFace tokenizer (passed to preprocess_data.py as --tokenizer-model)"
    )
    tokenizer_group.add_argument(
        "--tokenizer-type",
        type=str,
        default="HuggingFaceTokenizer",
        help="Tokenizer type for preprocess_data.py (default: HuggingFaceTokenizer)"
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for processed data"
    )

    # Runtime options
    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of worker processes per file (default: 24)"
    )
    runtime_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output already exists"
    )
    runtime_group.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for shuffling file list (default: 42)"
    )
    runtime_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without actually running"
    )

    # Extra arguments to pass to preprocess_data.py
    extra_group = parser.add_argument_group("Extra preprocess_data.py arguments")
    extra_group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append EOD token to documents"
    )
    extra_group.add_argument(
        "--split-sentences",
        action="store_true",
        help="Split documents into sentences"
    )
    extra_group.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Keep newlines when splitting sentences"
    )
    extra_group.add_argument(
        "--lang",
        type=str,
        default="english",
        help="Language for sentence splitting (default: english)"
    )

    return parser.parse_args()


def get_rank_and_world_size() -> Tuple[int, int]:
    """Get rank and world size from environment variables (PAI style)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("RC_WORLD_SIZE", 1))
    return rank, world_size


def collect_files(args: argparse.Namespace) -> List[Tuple[str, Optional[str]]]:
    """
    Collect all input files based on arguments.
    
    Returns:
        List of (file_path, category) tuples. category is None if using --input-glob.
    """
    file_list: List[Tuple[str, Optional[str]]] = []

    if args.input_glob:
        # Use glob pattern directly
        matched_files = glob.glob(args.input_glob, recursive=True)
        for f in matched_files:
            if f.endswith(".jsonl") or f.endswith(".jsonl.gz"):
                file_list.append((f, None))
    elif args.data_root and args.category:
        # Scan categories under data root
        for cat in args.category:
            cat_path = os.path.join(args.data_root, cat)
            if not os.path.isdir(cat_path):
                print(f"Warning: Category directory not found: {cat_path}", file=sys.stderr)
                continue
            for filename in os.listdir(cat_path):
                if filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"):
                    file_path = os.path.join(cat_path, filename)
                    file_list.append((file_path, cat))
    elif args.data_root:
        # Scan all .jsonl files under data root (recursively)
        for root, _, files in os.walk(args.data_root):
            for filename in files:
                if filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"):
                    file_path = os.path.join(root, filename)
                    # Use relative path as category
                    rel_path = os.path.relpath(root, args.data_root)
                    file_list.append((file_path, rel_path if rel_path != "." else None))
    else:
        raise ValueError(
            "Must specify either --input-glob or --data-root (optionally with --category)"
        )

    # Sort for deterministic ordering before shuffle
    file_list.sort(key=lambda x: x[0])
    return file_list


def distribute_files(
    file_list: List[Tuple[str, Optional[str]]],
    rank: int,
    world_size: int,
    shuffle_seed: int
) -> List[Tuple[str, Optional[str]]]:
    """
    Distribute files across ranks using fixed-seed shuffle for load balancing.
    
    All ranks use the same seed, so each rank computes the same shuffle order
    and then takes its own slice.
    """
    # Shuffle with fixed seed (same across all ranks)
    random.seed(shuffle_seed)
    shuffled_list = file_list.copy()
    random.shuffle(shuffled_list)

    print(f"Total files: {len(shuffled_list)}, World size: {world_size}, Rank: {rank}")
    print(f"Shuffled file list (first 5): {[f[0] for f in shuffled_list[:5]]}")

    # Split using numpy.array_split (handles uneven division gracefully)
    indices = range(len(shuffled_list))
    split_indices = np.array_split(indices, world_size)
    rank_indices = list(split_indices[rank])

    rank_files = [shuffled_list[i] for i in rank_indices]
    print(f"Rank {rank} will process {len(rank_files)} files")

    return rank_files


def get_output_prefix(
    file_path: str,
    category: Optional[str],
    output_path: str
) -> str:
    """Generate output prefix for a given input file."""
    # Extract file basename without extension
    basename = os.path.basename(file_path)
    if basename.endswith(".jsonl.gz"):
        file_id = basename[:-9]  # Remove .jsonl.gz
    elif basename.endswith(".jsonl"):
        file_id = basename[:-6]  # Remove .jsonl
    else:
        file_id = basename

    if category:
        output_dir = os.path.join(output_path, category)
    else:
        output_dir = output_path

    # Always use {output_dir}/{file_id} without any version prefix.
    return os.path.join(output_dir, f"{file_id}")


def check_output_exists(output_prefix: str, json_keys: List[str]) -> bool:
    """Check if output files already exist for all json keys."""
    for key in json_keys:
        # Check for document-level output (most common)
        bin_file = f"{output_prefix}_{key}_document.bin"
        idx_file = f"{output_prefix}_{key}_document.idx"
        if os.path.exists(bin_file) and os.path.exists(idx_file):
            continue
        # Also check sentence-level output
        bin_file_sent = f"{output_prefix}_{key}_sentence.bin"
        idx_file_sent = f"{output_prefix}_{key}_sentence.idx"
        if os.path.exists(bin_file_sent) and os.path.exists(idx_file_sent):
            continue
        return False
    return True


def build_preprocess_command(
    input_file: str,
    output_prefix: str,
    args: argparse.Namespace
) -> List[str]:
    """Build the command to call preprocess_data.py."""
    # Find the path to preprocess_data.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_script = os.path.join(script_dir, "..", "preprocess_data.py")
    preprocess_script = os.path.abspath(preprocess_script)

    cmd = [
        sys.executable,
        preprocess_script,
        "--input", input_file,
        "--output-prefix", output_prefix,
        "--tokenizer-type", args.tokenizer_type,
        "--tokenizer-model", args.tokenizer_model,
        "--workers", str(args.workers),
        "--json-keys", *args.json_keys,
    ]

    if args.append_eod:
        cmd.append("--append-eod")

    if args.split_sentences:
        cmd.append("--split-sentences")

    if args.keep_newlines:
        cmd.append("--keep-newlines")

    if args.lang != "english":
        cmd.extend(["--lang", args.lang])

    return cmd


def process_file(
    file_info: Tuple[str, Optional[str]],
    args: argparse.Namespace,
    rank: int,
    file_idx: int,
    total_files: int
) -> bool:
    """
    Process a single file by calling preprocess_data.py.
    
    Returns:
        True if successful, False otherwise.
    """
    file_path, category = file_info
    output_prefix = get_output_prefix(
        file_path, category, args.output_path
    )

    # Create output directory if needed
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Check if output already exists
    if args.skip_existing and check_output_exists(output_prefix, args.json_keys):
        print(f"[Rank {rank}] [{file_idx+1}/{total_files}] Skipping (exists): {file_path}")
        return True

    cmd = build_preprocess_command(file_path, output_prefix, args)

    print(f"[Rank {rank}] [{file_idx+1}/{total_files}] Processing: {file_path}")
    print(f"  Output prefix: {output_prefix}")

    if args.dry_run:
        print(f"  Command: {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"[Rank {rank}] ERROR: Failed to process {file_path}", file=sys.stderr)
        return False

    return True


def main():
    args = parse_args()
    rank, world_size = get_rank_and_world_size()

    print(f"=" * 60)
    print(f"Multi-node Data Preprocessing")
    print(f"Rank: {rank}, World Size: {world_size}")
    print(f"=" * 60)

    # Collect all input files
    file_list = collect_files(args)
    if not file_list:
        print("No input files found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(file_list)} total files to process")

    # Distribute files to this rank
    rank_files = distribute_files(file_list, rank, world_size, args.shuffle_seed)

    if not rank_files:
        print(f"[Rank {rank}] No files assigned to this rank")
        return

    print(f"\n[Rank {rank}] Files to process:")
    for i, (f, cat) in enumerate(rank_files):
        print(f"  {i+1}. {f} (category: {cat})")
    print()

    # Process each file
    success_count = 0
    fail_count = 0

    for idx, file_info in enumerate(tqdm(rank_files, desc=f"Rank {rank}")):
        success = process_file(file_info, args, rank, idx, len(rank_files))
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n[Rank {rank}] Completed: {success_count} success, {fail_count} failed")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
