#!/usr/bin/env python
"""Benchmark storage vs. EPE loss for different optical flow compression formats.

Loads float16 flow_raw.npz outputs produced by the benchmark pipeline (FlowSeek by
default), encodes each clip with every format, decodes back, and reports endpoint
error (EPE) and storage size.

Formats:
  original_lossless    -- the raw .npz (float16, no codec compression)
  rgb_symmetric        -- (dx, dy) packed into R+G of an H.264 MP4; max_abs sidecar
  vertical_no_range    -- (dx, dy) stacked vertically in a grayscale H.264 MP4,
                          clipped to +-125 (no sidecar, range is baked in)
  vertical_with_range  -- same vertical stack but with a per-file max_abs sidecar
"""

import argparse
import csv
import gc
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import imageio.v2 as imageio


CLIP_RANGE = 125.0


# ── I/O helpers ───────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def load_raw(path: Path) -> np.ndarray:
    with np.load(path) as f:
        return f["flows"].astype(np.float32)


def write_mp4(frames: np.ndarray, path: Path) -> None:
    """Write [T, H, W, 3] uint8 RGB frames as H.264 MP4."""
    with imageio.get_writer(str(path), fps=24, codec="libx264", macro_block_size=None) as w:
        for frame in frames:
            w.append_data(frame)


def ffmpeg_decode(path: Path, height: int, width: int) -> np.ndarray:
    """Decode video to [T, H, W, 3] uint8 RGB via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


# ── Metrics ───────────────────────────────────────────────────────────────────

def epe_stats(ref: np.ndarray, rec: np.ndarray) -> Dict:
    n = min(len(ref), len(rec))
    epe = np.linalg.norm(ref[:n] - rec[:n], axis=-1)
    return {
        "mean_epe": float(np.mean(epe)),
        "median_epe": float(np.median(epe)),
        "p95_epe": float(np.percentile(epe, 95)),
        "p99_epe": float(np.percentile(epe, 99)),
        "max_epe": float(np.max(epe)),
        "compared_frames": int(n),
    }


# ── Compression formats ───────────────────────────────────────────────────────

def encode_rgb_symmetric(flow: np.ndarray, tmp: Path) -> Tuple[np.ndarray, int]:
    """Pack (dx, dy) into the R and G channels of an H.264 MP4.
    Saves per-file max_abs as a small .metadata.npz sidecar."""
    max_abs = float(np.abs(flow).max()) or 1.0
    pixel = np.clip((flow / max_abs + 1.0) * 127.5, 0, 255).astype(np.uint8)
    rgb = np.zeros((*pixel.shape[:3], 3), dtype=np.uint8)
    rgb[..., :2] = pixel

    vid, meta = tmp / "rgb.mp4", tmp / "rgb.metadata.npz"
    write_mp4(rgb, vid)
    np.savez(meta, max_abs_flow=np.float32(max_abs))

    decoded = ffmpeg_decode(vid, flow.shape[1], flow.shape[2])
    rec = (decoded[..., :2].astype(np.float32) / 127.5 - 1.0) * max_abs
    return rec, vid.stat().st_size + meta.stat().st_size


def encode_vertical_no_range(flow: np.ndarray, tmp: Path) -> Tuple[np.ndarray, int]:
    """Stack dx (top) and dy (bottom) in a grayscale H.264 MP4.
    Clips values to +-CLIP_RANGE; no metadata saved."""
    clipped = np.clip(flow, -CLIP_RANGE, CLIP_RANGE)
    pixel = np.clip((clipped / CLIP_RANGE + 1.0) * 127.5, 0, 255).astype(np.uint8)
    gray = np.concatenate([pixel[..., 0], pixel[..., 1]], axis=1)  # [T, 2H, W]
    rgb = np.stack([gray, gray, gray], axis=-1)

    vid = tmp / "vert_no_range.mp4"
    write_mp4(rgb, vid)

    T, H, W = flow.shape[:3]
    frames = ffmpeg_decode(vid, H * 2, W)[..., 0]  # [T, 2H, W] — take R channel
    dx = frames[:, :H, :].astype(np.float32)
    dy = frames[:, H:, :].astype(np.float32)
    rec = (np.stack([dx, dy], axis=-1) / 127.5 - 1.0) * CLIP_RANGE
    return rec, vid.stat().st_size


def encode_vertical_with_range(flow: np.ndarray, tmp: Path) -> Tuple[np.ndarray, int]:
    """Stack dx (top) and dy (bottom) in a grayscale H.264 MP4.
    Uses per-file max_abs saved as a .metadata.npz sidecar."""
    max_abs = float(np.abs(flow).max()) or 1.0
    pixel = np.clip((flow / max_abs + 1.0) * 127.5, 0, 255).astype(np.uint8)
    gray = np.concatenate([pixel[..., 0], pixel[..., 1]], axis=1)  # [T, 2H, W]
    rgb = np.stack([gray, gray, gray], axis=-1)

    vid, meta = tmp / "vert_range.mp4", tmp / "vert_range.metadata.npz"
    write_mp4(rgb, vid)
    np.savez(meta, max_abs_flow=np.float32(max_abs))

    T, H, W = flow.shape[:3]
    frames = ffmpeg_decode(vid, H * 2, W)[..., 0]  # [T, 2H, W]
    dx = frames[:, :H, :].astype(np.float32)
    dy = frames[:, H:, :].astype(np.float32)
    rec = (np.stack([dx, dy], axis=-1) / 127.5 - 1.0) * max_abs
    return rec, vid.stat().st_size + meta.stat().st_size


METHODS = [
    ("rgb_symmetric", encode_rgb_symmetric),
    ("vertical_no_range", encode_vertical_no_range),
    ("vertical_with_range", encode_vertical_with_range),
]


# ── Per-sample benchmark ──────────────────────────────────────────────────────

def benchmark_sample(raw_path: Path) -> List[Dict]:
    flow = load_raw(raw_path)
    original_bytes = raw_path.stat().st_size
    sample = f"{raw_path.parent.parent.name}/{raw_path.parent.name}"
    rows = []

    rows.append({
        "sample": sample,
        "technique": "original_lossless",
        "mean_epe": 0.0, "median_epe": 0.0,
        "p95_epe": 0.0, "p99_epe": 0.0, "max_epe": 0.0,
        "storage_bytes": original_bytes,
        "storage_mib": round(original_bytes / 1024 ** 2, 4),
        "compression_ratio": 1.0,
        "compared_frames": int(flow.shape[0]),
    })

    with tempfile.TemporaryDirectory() as tmp_str:
        for name, fn in METHODS:
            rec, storage = fn(flow, Path(tmp_str))
            stats = epe_stats(flow, rec)
            rows.append({
                "sample": sample,
                "technique": name,
                **stats,
                "storage_bytes": storage,
                "storage_mib": round(storage / 1024 ** 2, 4),
                "compression_ratio": round(original_bytes / storage, 4) if storage else float("inf"),
            })
            del rec
            gc.collect()

    del flow
    gc.collect()
    return rows


# ── Aggregation & output ──────────────────────────────────────────────────────

def summarize(rows: List[Dict]) -> Dict:
    summary = {}
    for technique in sorted({r["technique"] for r in rows}):
        sel = [r for r in rows if r["technique"] == technique]
        total_bytes = sum(r["storage_bytes"] for r in sel)
        summary[technique] = {
            "files": len(sel),
            "total_mib": round(total_bytes / 1024 ** 2, 2),
            "mean_epe": round(float(np.mean([r["mean_epe"] for r in sel])), 6),
            "p95_epe": round(float(np.mean([r["p95_epe"] for r in sel])), 6),
            "p99_epe": round(float(np.mean([r["p99_epe"] for r in sel])), 6),
            "mean_compression_ratio": round(float(np.mean([r["compression_ratio"] for r in sel])), 3),
        }
    return summary


def write_csv(rows: List[Dict], path: Path) -> None:
    fieldnames = [
        "sample", "technique",
        "mean_epe", "median_epe", "p95_epe", "p99_epe", "max_epe",
        "storage_bytes", "storage_mib", "compression_ratio", "compared_frames",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def print_summary(summary: Dict) -> None:
    print("\nSummary")
    for technique, stats in summary.items():
        print(
            f"  {technique}: files={stats['files']} mib={stats['total_mib']:.1f} "
            f"mean_epe={stats['mean_epe']:.6f} p95={stats['p95_epe']:.6f} "
            f"ratio={stats['mean_compression_ratio']:.2f}x"
        )


def main() -> None:
    check_ffmpeg()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("/capstor/scratch/cscs/mlopezescoriza/dataset_benchmark_optical_flow/results/FlowSeek"),
        help="Directory containing category/stem/flow_raw.npz files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples.")
    parser.add_argument("--csv", type=Path, default=Path("compression_benchmark.csv"))
    parser.add_argument("--json", type=Path, default=Path("compression_benchmark_summary.json"))
    args = parser.parse_args()

    raw_paths = sorted(args.results_dir.rglob("flow_raw.npz"))
    if args.limit:
        raw_paths = raw_paths[: args.limit]
    if not raw_paths:
        raise FileNotFoundError(f"No flow_raw.npz files found under {args.results_dir}")

    all_rows: List[Dict] = []
    for i, path in enumerate(raw_paths, 1):
        sample = f"{path.parent.parent.name}/{path.parent.name}"
        print(f"[{i}/{len(raw_paths)}] {sample}", flush=True)
        try:
            all_rows.extend(benchmark_sample(path))
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)

    write_csv(all_rows, args.csv)
    summary = summarize(all_rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(f"\nCSV:  {args.csv}")
    print(f"JSON: {args.json}")


if __name__ == "__main__":
    main()
