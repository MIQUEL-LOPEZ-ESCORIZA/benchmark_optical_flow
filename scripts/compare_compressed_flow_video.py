#!/usr/bin/env python
"""Compare raw benchmark optical flow with the RGB-compressed pipeline output."""

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np


def flow_to_rgb(flow: np.ndarray, flow_max_radius: float) -> np.ndarray:
    transitions = (15, 6, 4, 11, 13, 6)
    hues = np.array(
        [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0]],
        dtype=np.float32,
    )
    wheel = np.concatenate(
        [np.linspace(hues[i], hues[i + 1], n, endpoint=False) for i, n in enumerate(transitions)],
        axis=0,
    )

    flow = flow.astype(np.float32, copy=False)
    radius = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    if flow_max_radius > 0:
        radius = radius / flow_max_radius

    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    angle = angle * ((len(wheel) - 1) / (2 * np.pi))
    wheel = np.vstack([wheel, wheel[0]])
    lo = np.floor(angle).astype(np.int32)
    hi = np.ceil(angle).astype(np.int32)
    frac = (angle - lo)[..., None]
    hue = wheel[lo] * (1 - frac) + wheel[hi] * frac

    rgb = 255.0 - radius[..., None] * (255.0 - hue)
    too_large = radius > 1
    rgb[too_large] = hue[too_large] / radius[too_large][..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def read_video(path: Path) -> np.ndarray:
    return np.stack([frame[:, :, :3] for frame in imageio.get_reader(path)], axis=0)


def decode_rgb_symmetric(video_path: Path) -> np.ndarray:
    metadata_path = video_path.with_suffix(".metadata.npz")
    metadata = np.load(metadata_path)
    max_abs = float(metadata["max_abs_flow"])
    rgb = read_video(video_path).astype(np.float32)
    return (rgb[..., :2] / 127.5 - 1.0) * max_abs


def resize_flow(flow: np.ndarray, height: int, width: int) -> np.ndarray:
    if flow.shape[1:3] == (height, width):
        return flow
    y_scale = height / flow.shape[1]
    x_scale = width / flow.shape[2]
    resized = []
    for frame in flow:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frame[..., 0] *= x_scale
        frame[..., 1] *= y_scale
        resized.append(frame)
    return np.stack(resized, axis=0)


def find_compressed_video(compressed_dir: Path, stem: str) -> Optional[Path]:
    shard = stem.split("_")[-1]
    candidates = [
        compressed_dir / shard / f"{stem}.mp4",
        compressed_dir / shard.zfill(5) / f"{stem}.mp4",
        compressed_dir / f"{stem}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(compressed_dir.rglob(f"{stem}.mp4"))
    return matches[0] if matches else None


def make_comparison(raw_flow: np.ndarray, compressed_flow: np.ndarray) -> np.ndarray:
    n = min(len(raw_flow), len(compressed_flow))
    raw_flow = raw_flow[:n]
    compressed_flow = resize_flow(compressed_flow[:n], raw_flow.shape[1], raw_flow.shape[2])
    radius = np.sqrt(np.concatenate([raw_flow, compressed_flow], axis=0)[..., 0] ** 2 + np.concatenate([raw_flow, compressed_flow], axis=0)[..., 1] ** 2)
    flow_max_radius = float(np.percentile(radius, 99.5))

    frames = []
    for raw_frame, compressed_frame in zip(raw_flow, compressed_flow):
        raw_rgb = flow_to_rgb(raw_frame, flow_max_radius)
        compressed_rgb = flow_to_rgb(compressed_frame, flow_max_radius)
        diff = np.linalg.norm(raw_frame - compressed_frame, axis=-1)
        diff = np.clip(diff / max(flow_max_radius, 1e-6), 0, 1)
        diff_rgb = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)[:, :, ::-1]
        panel = np.concatenate([raw_rgb, compressed_rgb, diff_rgb], axis=1)
        width = raw_rgb.shape[1]
        for x, label in [(8, "benchmark raw"), (width + 8, "decoded compressed"), (2 * width + 8, "endpoint diff")]:
            cv2.putText(panel, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        frames.append(panel)
    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed-dir", type=Path, default=Path("/capstor/scratch/cscs/mlopezescoriza/sample_dataset_video/subset-500/optical_flow"))
    parser.add_argument("--benchmark-results", type=Path, default=Path("/capstor/scratch/cscs/mlopezescoriza/dataset_benchmark_optical_flow/results"))
    parser.add_argument("--model", default="FlowSeek")
    parser.add_argument("--out-dir", type=Path, default=Path("/capstor/scratch/cscs/mlopezescoriza/dataset_benchmark_optical_flow/compression_compare"))
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.benchmark_results / "manifest.tsv"
    made = 0

    with manifest_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            category = row["category"]
            stem = row["video_stem"]
            raw_path = args.benchmark_results / args.model / category / stem / "flow_raw.npz"
            compressed_path = find_compressed_video(args.compressed_dir, stem)
            if not raw_path.exists() or compressed_path is None:
                continue

            out_path = args.out_dir / args.model / category / f"{stem}_raw_vs_compressed.mp4"
            if out_path.exists() and not args.overwrite:
                made += 1
                continue

            raw_flow = np.load(raw_path)["flows"].astype(np.float32)
            compressed_flow = decode_rgb_symmetric(compressed_path).astype(np.float32)
            comparison = make_comparison(raw_flow, compressed_flow)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fps = float(row["fps"]) if row.get("fps") else 24.0
            imageio.mimwrite(out_path, comparison, fps=fps, codec="libx264", macro_block_size=None)
            print(out_path)
            made += 1
            if made >= args.max_videos:
                break

    print(f"Wrote/found {made} comparison videos in {args.out_dir}")


if __name__ == "__main__":
    main()
