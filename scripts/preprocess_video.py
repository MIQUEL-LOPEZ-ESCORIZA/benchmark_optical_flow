#!/usr/bin/env python
"""Preprocess a video for shared optical-flow benchmark inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def read_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        return 0.0
    return float(fps)


def preprocess(input_path: str, output_path: str, target_fps: float, frame_stride: int) -> float:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 24.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Cannot read video dimensions: {input_path}")

    if frame_stride > 1:
        output_fps = source_fps / frame_stride
        keep_frame = lambda idx: idx % frame_stride == 0
    elif target_fps > 0:
        output_fps = target_fps
        step = source_fps / target_fps
        next_idx = 0.0

        def keep_frame(idx: int) -> bool:
            nonlocal next_idx
            if idx + 1e-6 >= next_idx:
                next_idx += step
                return True
            return False
    else:
        output_fps = source_fps
        keep_frame = lambda idx: True

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    written = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if keep_frame(idx):
            writer.write(frame)
            written += 1
        idx += 1

    cap.release()
    writer.release()

    if frame_count > 0 and written < 2:
        raise ValueError(f"Only wrote {written} frames to {output_path}; need at least 2")

    return output_fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--target_fps", type=float, default=0.0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--print_fps_only", action="store_true")
    args = parser.parse_args()

    if args.print_fps_only:
        print(read_fps(args.input))
        return

    if not args.output:
        raise ValueError("--output is required unless --print_fps_only is set")
    output_fps = preprocess(args.input, args.output, args.target_fps, args.frame_stride)
    print(output_fps)


if __name__ == "__main__":
    main()
