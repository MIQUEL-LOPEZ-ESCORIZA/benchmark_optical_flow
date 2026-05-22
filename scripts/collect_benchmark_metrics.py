#!/usr/bin/env python
"""Wrap benchmark runs with GPU metric collection and aggregate CSV results."""

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_metadata(path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if not path.exists():
        return metadata
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


class GPUSampler(threading.Thread):
    def __init__(self, interval_seconds: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval_seconds = interval_seconds
        self.samples: List[float] = []
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._query_gpu_utilization()
            if sample is not None:
                self.samples.append(sample)
            self._stop_event.wait(self.interval_seconds)

    def _query_gpu_utilization(self) -> Optional[float]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

        values = []
        for line in result.stdout.splitlines():
            value = safe_float(line.strip())
            if value is not None:
                values.append(value)
        if not values:
            return None
        return max(values)


def build_metrics_record(
    args: argparse.Namespace, return_code: int, runtime_seconds: float, gpu_samples: List[float]
) -> Dict[str, Any]:
    metadata = parse_metadata(Path(args.output_dir) / "metadata.txt")
    input_size_bytes = Path(args.input_video).stat().st_size if Path(args.input_video).exists() else 0
    output_size_bytes = compute_dir_size(Path(args.output_dir))

    output_fps = safe_float(metadata.get("output_fps"))
    num_input_frames = safe_int(metadata.get("num_input_frames"))
    input_duration_seconds = None
    if output_fps and output_fps > 0 and num_input_frames and num_input_frames > 0:
        input_duration_seconds = num_input_frames / output_fps

    relative_storage = None
    if input_size_bytes > 0:
        relative_storage = output_size_bytes / input_size_bytes

    gpu_hours_per_input_hour = None
    if input_duration_seconds and input_duration_seconds > 0:
        gpu_hours_per_input_hour = runtime_seconds / input_duration_seconds

    avg_gpu_utilization = None
    if gpu_samples:
        avg_gpu_utilization = sum(gpu_samples) / len(gpu_samples)

    return {
        "model": args.model,
        "category": args.category,
        "video_stem": args.video_stem,
        "input_video": args.input_video,
        "output_dir": args.output_dir,
        "log_path": args.log_path,
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
        "runtime_seconds": runtime_seconds,
        "input_duration_seconds": input_duration_seconds,
        "gpu_hours_per_input_hour": gpu_hours_per_input_hour,
        "average_gpu_utilization_percent": avg_gpu_utilization,
        "gpu_utilization_sample_count": len(gpu_samples),
        "input_size_bytes": input_size_bytes,
        "output_size_bytes": output_size_bytes,
        "relative_storage": relative_storage,
        "metadata": metadata,
    }


def command_run(args: argparse.Namespace) -> int:
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = GPUSampler(interval_seconds=args.gpu_sample_interval_seconds)
    start_time = time.time()
    process = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    sampler.start()

    with log_path.open("w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)

    return_code = process.wait()
    runtime_seconds = time.time() - start_time
    sampler.stop()
    sampler.join(timeout=2)

    metrics = build_metrics_record(args, return_code, runtime_seconds, sampler.samples)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return return_code


def load_manifest(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows[(row["category"], row["video_stem"])] = row
    return rows


def format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def command_aggregate(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    manifest_rows = load_manifest(Path(args.manifest))
    metric_paths = sorted(results_dir.glob("*/**/benchmark_metrics.json"))

    fieldnames = [
        "row_type",
        "model",
        "category",
        "video_stem",
        "input_video",
        "preprocessed_video",
        "status",
        "runtime_seconds",
        "input_duration_seconds",
        "gpu_hours_per_input_hour",
        "average_gpu_utilization_percent",
        "input_size_bytes",
        "output_size_bytes",
        "relative_storage",
        "log_path",
        "output_dir",
    ]

    per_video_rows: List[Dict[str, Any]] = []
    aggregates: Dict[str, Dict[str, float]] = {}

    for metric_path in metric_paths:
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        manifest_row = manifest_rows.get((payload["category"], payload["video_stem"]), {})

        row = {
            "row_type": "per_video",
            "model": payload.get("model", ""),
            "category": payload.get("category", ""),
            "video_stem": payload.get("video_stem", ""),
            "input_video": manifest_row.get("input_video", payload.get("input_video", "")),
            "preprocessed_video": manifest_row.get("preprocessed_video", ""),
            "status": payload.get("status", ""),
            "runtime_seconds": format_number(payload.get("runtime_seconds")),
            "input_duration_seconds": format_number(payload.get("input_duration_seconds")),
            "gpu_hours_per_input_hour": format_number(payload.get("gpu_hours_per_input_hour")),
            "average_gpu_utilization_percent": format_number(payload.get("average_gpu_utilization_percent")),
            "input_size_bytes": format_number(payload.get("input_size_bytes")),
            "output_size_bytes": format_number(payload.get("output_size_bytes")),
            "relative_storage": format_number(payload.get("relative_storage")),
            "log_path": payload.get("log_path", ""),
            "output_dir": payload.get("output_dir", ""),
        }
        per_video_rows.append(row)

        if payload.get("status") != "ok":
            continue

        model = payload["model"]
        aggregate = aggregates.setdefault(
            model,
            {
                "runtime_seconds": 0.0,
                "input_duration_seconds": 0.0,
                "weighted_gpu_util_sum": 0.0,
                "input_size_bytes": 0.0,
                "output_size_bytes": 0.0,
            },
        )
        runtime_seconds = safe_float(payload.get("runtime_seconds")) or 0.0
        input_duration_seconds = safe_float(payload.get("input_duration_seconds")) or 0.0
        gpu_util = safe_float(payload.get("average_gpu_utilization_percent"))
        input_size_bytes = safe_float(payload.get("input_size_bytes")) or 0.0
        output_size_bytes = safe_float(payload.get("output_size_bytes")) or 0.0

        aggregate["runtime_seconds"] += runtime_seconds
        aggregate["input_duration_seconds"] += input_duration_seconds
        aggregate["input_size_bytes"] += input_size_bytes
        aggregate["output_size_bytes"] += output_size_bytes
        if gpu_util is not None:
            aggregate["weighted_gpu_util_sum"] += gpu_util * runtime_seconds

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_video_rows:
            writer.writerow(row)
        for model in sorted(aggregates):
            aggregate = aggregates[model]
            runtime_seconds = aggregate["runtime_seconds"]
            input_duration_seconds = aggregate["input_duration_seconds"]
            input_size_bytes = aggregate["input_size_bytes"]
            output_size_bytes = aggregate["output_size_bytes"]
            writer.writerow(
                {
                    "row_type": "aggregate",
                    "model": model,
                    "category": "__all__",
                    "video_stem": "__all_videos__",
                    "input_video": "",
                    "preprocessed_video": "",
                    "status": "ok",
                    "runtime_seconds": format_number(runtime_seconds),
                    "input_duration_seconds": format_number(input_duration_seconds),
                    "gpu_hours_per_input_hour": format_number(
                        runtime_seconds / input_duration_seconds if input_duration_seconds > 0 else None
                    ),
                    "average_gpu_utilization_percent": format_number(
                        aggregate["weighted_gpu_util_sum"] / runtime_seconds if runtime_seconds > 0 else None
                    ),
                    "input_size_bytes": format_number(input_size_bytes),
                    "output_size_bytes": format_number(output_size_bytes),
                    "relative_storage": format_number(
                        output_size_bytes / input_size_bytes if input_size_bytes > 0 else None
                    ),
                    "log_path": "",
                    "output_dir": "",
                }
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--log-path", required=True)
    run_parser.add_argument("--metrics-path", required=True)
    run_parser.add_argument("--input-video", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--category", required=True)
    run_parser.add_argument("--video-stem", required=True)
    run_parser.add_argument("--gpu-sample-interval-seconds", type=float, default=1.0)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=command_run)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--results-dir", required=True)
    aggregate_parser.add_argument("--manifest", required=True)
    aggregate_parser.add_argument("--output-csv", required=True)
    aggregate_parser.set_defaults(func=command_aggregate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command_name", None):
        parser.error("a subcommand is required")
    if args.command_name == "run":
        if not args.command or args.command[0] != "--":
            parser.error("run requires a command after --")
        args.command = args.command[1:]
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
