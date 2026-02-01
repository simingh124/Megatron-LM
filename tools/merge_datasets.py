import os
import sys
import json
import argparse
from typing import Iterable, List, Optional

from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory containing all document files to merge",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    group = parser.add_argument_group(title="miscellaneous")
    group.add_argument(
        "--multimodal",
        action="store_true",
        help="Whether the datasets are assumed to be multimodal"
    )
    group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-shard progress output.",
    )

    args = parser.parse_args()

    assert os.path.isdir(
        args.input
    ), f"ERROR: {args.input} is not a directory or does not exist"

    assert os.path.isdir(
        os.path.dirname(args.output_prefix)
    ), f"ERROR: {os.path.dirname(args.output_prefix)} is not a directory or does not exist"

    return args


def _iter_with_progress(prefixes: List[str], desc: str, disable: bool) -> Iterable[str]:
    """
    Yield prefixes and update progress once per prefix.

    Progress is implemented with tqdm (required).
    """
    if disable:
        return prefixes

    return tqdm(prefixes, desc=desc, dynamic_ncols=True)


def merge_dir(
    input_dir: str,
    output_prefix: str,
    multimodal: bool,
    no_progress: bool,
    desc: Optional[str] = None,
) -> int:
    """
    Merge all shard prefixes found in input_dir into output_prefix.

    Returns:
        Number of shard prefixes merged.
    """
    # Always overwrite existing output artifacts to avoid resuming from partial results.
    out_bin = get_bin_path(output_prefix)
    out_idx = get_idx_path(output_prefix)
    if os.path.exists(out_bin):
        os.remove(out_bin)
    if os.path.exists(out_idx):
        os.remove(out_idx)

    prefixes = set()
    for basename in os.listdir(input_dir):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(input_dir, basename)):
            continue

        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(
            os.path.join(input_dir, prefix) + ext_pair
        ), f"ERROR: {ext_pair} file not provided for {os.path.join(input_dir, prefix)}"

        prefixes.add(prefix)

    builder = None
    sorted_prefixes = sorted(prefixes)
    if desc is None:
        desc = os.path.basename(output_prefix)

    for prefix in _iter_with_progress(sorted_prefixes, desc=desc, disable=no_progress):
        if builder is None:
            dataset = IndexedDataset(os.path.join(input_dir, prefix), multimodal=multimodal)
            builder = IndexedDatasetBuilder(
                get_bin_path(output_prefix), dtype=dataset.index.dtype, multimodal=multimodal
            )
            del dataset

        builder.add_index(os.path.join(input_dir, prefix))

    builder.finalize(get_idx_path(output_prefix))
    return len(sorted_prefixes)


def main():
    args = get_args()

    merge_dir(
        input_dir=args.input,
        output_prefix=args.output_prefix,
        multimodal=args.multimodal,
        no_progress=args.no_progress,
        desc=os.path.basename(args.output_prefix),
    )


if __name__ == '__main__':

    main()
