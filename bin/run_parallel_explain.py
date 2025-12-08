#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List


def chunk_indices(total: int, chunk_size: int) -> Iterable[List[int]]:
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield list(range(start, end))


def build_command(
    script_path: Path,
    dataset_name: str,
    exp_name: str,
    job_id: str,
    indices: List[int],
) -> List[str]:
    indices_arg = ",".join(str(idx) for idx in indices)
    return [
        sys.executable,
        str(script_path),
        "--dataset_name",
        dataset_name,
        "--exp_name",
        exp_name,
        "--job_id",
        job_id,
        "--dataset_indices",
        indices_arg,
    ]


def submit_job(command: List[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=False, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run explain_driver.py in parallel over index subsets."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Lung",
        help="Dataset name passed to explain_driver.py (default: Lung).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name passed to explain_driver.py.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        required=True,
        help="Job ID passed to explain_driver.py.",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=416,
        help="Total number of samples in the dataset (default: 416).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of samples per explain_driver.py invocation (default: 10).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=40,
        help="Maximum number of parallel explain_driver.py processes (default: 40).",
    )
    parser.add_argument(
        "--num_threads_per_job",
        type=int,
        default=2,
        help="Maximum number of threads each explain_driver.py process may use (default: 2).",
    )

    args, passthrough = parser.parse_known_args()

    script_path = (Path(__file__).resolve().parent / "explain_driver_classification.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Unable to locate explain_driver_classification.py at {script_path}")

    commands = [
        build_command(
            script_path=script_path,
            dataset_name=args.dataset_name,
            exp_name=args.exp_name,
            job_id=args.job_id,
            indices=indices,
        )
        + passthrough
        for indices in chunk_indices(args.total_samples, args.chunk_size)
    ]

    env = os.environ.copy()
    if args.num_threads_per_job is not None:
        thread_val = str(args.num_threads_per_job)
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TORCH_NUM_THREADS",
        ):
            env[var] = thread_val

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_cmd = {executor.submit(submit_job, cmd, env): cmd for cmd in commands}
        for future in as_completed(future_to_cmd):
            cmd = future_to_cmd[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - safety net
                print(f"Command failed before completion: {' '.join(cmd)}", file=sys.stderr)
                print(f"Raised exception: {exc}", file=sys.stderr)
                results.append(False)
            else:
                if result.returncode != 0:
                    print(
                        f"Command exited with code {result.returncode}: {' '.join(cmd)}",
                        file=sys.stderr,
                    )
                    results.append(False)
                else:
                    results.append(True)

    if results and not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()

# python run_parallel_explain.py --exp_name "Lung" --job_id Lung --max_workers 40