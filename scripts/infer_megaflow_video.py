#!/usr/bin/env python
"""Run MegaFlow on one benchmark video and save visual plus raw outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from megaflow.model import MegaFlow
from megaflow.utils.flow_viz import flow_to_image


def calculate_dynamic_size(orig_h: int, orig_w: int, target_fix_width: int, patch_size: int = 14) -> tuple[int, int]:
    if orig_w >= orig_h:
        new_w = target_fix_width
        new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
    else:
        new_h = target_fix_width
        new_w = round(orig_w * (new_h / orig_h) / patch_size) * patch_size
    return int(new_h), int(new_w)


def read_video(video_path: str, fix_width: int | None) -> tuple[list[torch.Tensor], tuple[int, int], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 24.0

    frames = []
    native_size = None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        native_size = native_size or frame_rgb.shape[:2]
        if fix_width and fix_width > 0:
            new_h, new_w = calculate_dynamic_size(frame_rgb.shape[0], frame_rgb.shape[1], fix_width)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).float())
    cap.release()

    if native_size is None:
        raise ValueError(f"No frames found in {video_path}")
    return frames, native_size, float(fps)


@torch.inference_mode()
def infer(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, native_size, native_fps = read_video(args.input_path, args.fix_width)
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames in {args.input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MegaFlow.from_pretrained(args.model_name, device=device).eval()

    input_scene = torch.stack(frames, dim=0)[None]
    _, total_frames, _, height, width = input_scene.shape

    output_fps = args.output_fps if args.output_fps and args.output_fps > 0 else native_fps
    video_path = output_dir / "flow_viz.mp4"
    video_writer = imageio.get_writer(video_path, fps=output_fps, codec="libx264", macro_block_size=None)

    raw_flows = []
    try:
        for start in tqdm(range(0, total_frames - 1, args.window_size - 1), desc="MegaFlow windows"):
            end = min(start + args.window_size, total_frames)
            chunk = input_scene[:, start:end].to(device)
            compute_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device, dtype=compute_dtype, enabled=(device == "cuda")):
                result = model(chunk, num_reg_refine=args.iters)

            flow_pr = result["flow_preds"][-1]
            if args.restore_size:
                scaled_flow = F.interpolate(
                    flow_pr.reshape(-1, 2, height, width),
                    size=native_size,
                    mode="bilinear",
                    align_corners=True,
                )
                scale_y = native_size[0] / height
                scale_x = native_size[1] / width
                scaled_flow[:, 0, :, :] *= scale_x
                scaled_flow[:, 1, :, :] *= scale_y
                flow_pr = scaled_flow.reshape(*flow_pr.shape[:2], 2, *native_size)

            flows = flow_pr[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            for local_idx, flow in enumerate(flows):
                frame_idx = start + local_idx + 1
                flow_rgb = flow_to_image(flow, convert_to_bgr=False)
                video_writer.append_data(flow_rgb)
                raw_flows.append(flow.astype(np.float16))

                if args.save_frames:
                    stem = f"frame_{frame_idx:06d}"
                    viz_dir = output_dir / "frames" / "flow_viz"
                    flow_dir = output_dir / "frames" / "flow"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    flow_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(viz_dir / f"{stem}.png"), cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))
                    write_flo(flow_dir / f"{stem}.flo", flow)
    finally:
        video_writer.close()

    if args.save_raw_npz:
        np.savez_compressed(output_dir / "flow_raw.npz", flows=np.stack(raw_flows, axis=0))

    (output_dir / "metadata.txt").write_text(
        "\n".join(
            [
                f"input_path={args.input_path}",
                f"model={args.model_name}",
                f"native_fps={native_fps}",
                f"output_fps={output_fps}",
                f"fix_width={args.fix_width}",
                f"restore_size={args.restore_size}",
                f"window_size={args.window_size}",
                f"iters={args.iters}",
                f"num_input_frames={len(frames)}",
                f"num_flow_frames={len(raw_flows)}",
            ]
        )
        + "\n"
    )


def write_flo(path: Path, flow: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="megaflow-flow")
    parser.add_argument("--output_fps", type=float, default=0.0)
    parser.add_argument("--fix_width", type=int, default=952)
    parser.add_argument("--restore_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--no_save_raw_npz", action="store_false", dest="save_raw_npz")
    parser.set_defaults(save_raw_npz=True)
    infer(parser.parse_args())


if __name__ == "__main__":
    main()
