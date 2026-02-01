#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-PROCESS merge utility for Megatron-LM IndexedDataset shards.

This is the multiprocessing counterpart of `mp_merge_data.py` (thread-based).
It parallelizes by category/directory, while keeping each directory's merge
sequential (builder is not process-safe / not parallelizable per output).
"""

import argparse
import os
import sys
from multiprocessing import Pool
from typing import List, Tuple

try:
    from tqdm import tqdm  # type: ignore

    _HAVE_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAVE_TQDM = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from tools.merge_datasets import merge_dir  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-process merge for Megatron-LM indexed datasets")

    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory containing shards to merge (single merge mode).",
    )
    input_group.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing categories to merge (multi-merge mode).",
    )
    input_group.add_argument(
        "--category",
        nargs="+",
        default=[],
        help="One or more category subdirs under --data-root to merge in parallel.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output prefix for single merge mode (no suffix).",
    )
    output_group.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output root directory for multi-merge mode.",
    )
    output_group.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output file name: {category}_{tag}. If empty, output is {category}.",
    )

    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of processes used to merge multiple categories (default: 8).",
    )
    misc_group.add_argument(
        "--multimodal",
        action="store_true",
        help="Whether the datasets are assumed to be multimodal.",
    )
    misc_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (tqdm).",
    )

    args = parser.parse_args()

    single_mode = args.input is not None or args.output_prefix is not None
    multi_mode = args.data_root is not None or len(args.category) > 0 or args.output_path is not None

    if single_mode and multi_mode and args.input is not None and args.data_root is not None:
        raise ValueError(
            "Use either single merge mode (--input/--output-prefix) OR multi-merge mode (--data-root/--category/--output-path)."
        )

    if args.input is not None or args.output_prefix is not None:
        if not args.input or not args.output_prefix:
            raise ValueError("Single merge mode requires both --input and --output-prefix.")
    else:
        if not args.data_root or not args.output_path or not args.category:
            raise ValueError("Multi-merge mode requires --data-root, --output-path, and --category.")

    return args


def _list_prefixes_in_dir(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    entries = os.listdir(input_dir)
    bins = set()
    idxs = set()
    for name in entries:
        full = os.path.join(input_dir, name)
        if not os.path.isfile(full):
            continue
        if name.endswith(".bin"):
            bins.add(os.path.join(input_dir, name[:-4]))
        elif name.endswith(".idx"):
            idxs.add(os.path.join(input_dir, name[:-4]))
    return sorted(bins.intersection(idxs))


def merge_one_dir(input_dir: str, output_prefix: str, multimodal: bool) -> Tuple[str, int]:
    # Reuse the single-directory merge logic from tools/merge_datasets.py
    # so progress updates happen per shard in each process.
    n = merge_dir(
        input_dir=input_dir,
        output_prefix=output_prefix,
        multimodal=multimodal,
        no_progress=False,
        desc=os.path.basename(output_prefix),
    )
    return output_prefix, n


def _output_prefix_for_category(output_path: str, category: str, tag: str) -> str:
    name = category if not tag else f"{category}_{tag}"
    return os.path.join(output_path, name)


def _merge_job(args: Tuple[str, str, bool]) -> Tuple[str, int]:
    # Kept for backward compatibility with the pool interface.
    # The real work is done by `merge_one_dir()` below.
    in_dir, out_prefix, multimodal = args
    return merge_one_dir(in_dir, out_prefix, multimodal)


def main() -> None:
    args = _parse_args()
    enable_progress = _HAVE_TQDM and not args.no_progress

    # Single merge mode
    if args.input is not None:
        out, n = merge_one_dir(args.input, args.output_prefix, args.multimodal)
        print(f"Merged {n} shards -> {out}")
        return

    # Multi-merge mode (parallel by category)
    os.makedirs(args.output_path, exist_ok=True)
    jobs: List[Tuple[str, str, bool]] = []
    for c in args.category:
        in_dir = os.path.join(args.data_root, c)
        out_prefix = _output_prefix_for_category(args.output_path, c, args.tag)
        jobs.append((in_dir, out_prefix, args.multimodal))

    workers = max(1, int(args.workers))
    results: List[Tuple[str, int]] = []

    # Optional totals for progress bars
    total_jobs = len(jobs)
    total_shards = None
    if enable_progress:
        # Pre-count shards per job in the parent process (fast, metadata-only)
        # so we can show an overall shard progress bar (updated per completed job).
        total = 0
        for in_dir, _out_prefix, _mm in jobs:
            try:
                total += len(_list_prefixes_in_dir(in_dir))
            except Exception:
                # Keep going; the actual merge will raise with details.
                pass
        total_shards = total

    p_jobs = None
    p_shards = None
    if enable_progress:
        p_jobs = tqdm(total=total_jobs, desc="merge_all(categories)", dynamic_ncols=True)
        p_shards = tqdm(total=total_shards, desc="merge_all(shards)", dynamic_ncols=True) if total_shards else None

    with Pool(processes=workers) as pool:
        for out, n in pool.imap_unordered(_merge_job, jobs):
            results.append((out, n))
            if p_jobs is not None:
                p_jobs.update(1)
            if p_shards is not None:
                p_shards.update(n)
            print(f"[OK] {out} (shards: {n})")

    if p_jobs is not None:
        p_jobs.close()
    if p_shards is not None:
        p_shards.close()

    results.sort(key=lambda x: x[0])
    print(f"Done. Merged {len(results)} categories with {workers} processes.")


if __name__ == "__main__":
    main()

