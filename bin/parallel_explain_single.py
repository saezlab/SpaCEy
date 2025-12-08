#!/usr/bin/env python3
"""Dispatch single-sample explainer jobs across CPU cores.

This launcher submits multiple invocations of ``explain_single_sample.py`` so
that different dataset indices are processed in parallel. Each worker runs in
its own process, ensuring isolation and reproducibility.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence

from explain_single_sample import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_SCRIPT = (SCRIPT_DIR / "explain_single_sample.py").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-sample explainer across multiple cores."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., JacksonFischer, METABRIC, Lung).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment/model name.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        required=True,
        help="Model job identifier.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers to spawn (default: 4).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of sample indices to process. "
        "If omitted, a contiguous range is used.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (inclusive) for automatic range selection (default: 0).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index (exclusive) for automatic range selection. "
        "Defaults to the dataset length.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples processed from the selected range.",
    )
    parser.add_argument(
        "--edge_quantile",
        type=float,
        default=0.80,
        help="Edge quantile passed to the single-sample script (default: 0.80).",
    )
    parser.add_argument(
        "--num_hops",
        type=int,
        default=2,
        help="Hop count passed to the single-sample script (default: 2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Epoch count passed to the single-sample script (default: 200).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device forwarded to the single-sample script (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed forwarded to the single-sample script (default: 42).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting in the single-sample job.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def determine_indices(
    args: argparse.Namespace, dataset_length: int
) -> Sequence[int]:
    if args.indices is not None:
        return [idx for idx in args.indices if 0 <= idx < dataset_length]

    start = max(0, args.start_idx)
    end = args.end_idx if args.end_idx is not None else dataset_length
    end = min(end, dataset_length)

    if start >= end:
        raise ValueError(
            f"Invalid index range: start_idx={start} end_idx={end} for dataset length {dataset_length}."
        )

    selected = list(range(start, end))

    if args.max_samples is not None:
        selected = selected[: args.max_samples]

    return selected


def build_command(args: argparse.Namespace, sample_idx: int) -> List[str]:
    cmd = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--dataset_name",
        args.dataset_name,
        "--exp_name",
        args.exp_name,
        "--job_id",
        args.job_id,
        "--sample_idx",
        str(sample_idx),
        "--edge_quantile",
        f"{args.edge_quantile}",
        "--num_hops",
        f"{args.num_hops}",
        "--epochs",
        f"{args.epochs}",
        "--device",
        args.device,
        "--seed",
        f"{args.seed}",
    ]

    if args.plot:
        cmd.append("--plot")

    return cmd


def launch_command(cmd: Iterable[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    args = parse_args()

    if not SINGLE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Required script not found: {SINGLE_SCRIPT}"
        )

    dataset = load_dataset(args.dataset_name)
    dataset_length = len(dataset)

    indices = determine_indices(args, dataset_length)
    if not indices:
        print("No indices selected. Nothing to do.")
        return 0

    print(f"Dispatching {len(indices)} sample(s) across {args.num_workers} worker(s).")

    commands = [(idx, build_command(args, idx)) for idx in indices]

    if args.dry_run:
        for idx, cmd in commands:
            print(f"[DRY RUN] idx={idx}: {' '.join(cmd)}")
        return 0

    successes: List[int] = []
    failures: List[tuple[int, int]] = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_idx = {
            executor.submit(launch_command, cmd): idx for idx, cmd in commands
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result.returncode == 0:
                successes.append(idx)
                if result.stdout:
                    print(result.stdout.strip())
            else:
                failures.append((idx, result.returncode))
                print(
                    f"[ERROR] Sample idx {idx} failed with return code {result.returncode}",
                    file=sys.stderr,
                )
                if result.stdout:
                    print(result.stdout.strip(), file=sys.stderr)
                if result.stderr:
                    print(result.stderr.strip(), file=sys.stderr)

    print(
        f"Completed. Successes: {len(successes)} | Failures: {len(failures)}"
    )

    if failures:
        failed_indices = ", ".join(f"{idx}(rc={rc})" for idx, rc in failures)
        print(f"Failed sample indices: {failed_indices}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





