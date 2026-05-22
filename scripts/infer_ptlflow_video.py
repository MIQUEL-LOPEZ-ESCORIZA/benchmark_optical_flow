#!/usr/bin/env python
"""Run one PTLFlow model on one benchmark video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2 as cv
import decord
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from jsonargparse import ArgumentParser, Namespace
from tqdm import tqdm

import ptlflow
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.flow_utils import flow_to_rgb, flow_write
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.registry import RegisteredModel


def _base_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--extract_fps", type=float, default=0.0)
    parser.add_argument("--output_fps", type=float, default=0.0)
    parser.add_argument("--input_size", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--scale_factor", type=float, default=None)
    parser.add_argument("--flow_format", choices=["flo", "png"], default="flo")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--no_save_raw_npz", action="store_false", dest="save_raw_npz")
    parser.add_argument("--fp16", action="store_true")
    parser.set_defaults(save_raw_npz=True)
    return parser


def _frames_from_video(video_path: str, target_fps: float) -> tuple[list[np.ndarray], float]:
    decord.bridge.set_bridge("native")
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = float(vr.get_avg_fps())
    if target_fps and target_fps > 0:
        step = native_fps / target_fps
        indices = []
        idx = 0.0
        while int(idx) < len(vr):
            indices.append(int(idx))
            idx += step
    else:
        indices = list(range(len(vr)))

    frames = []
    for idx in indices:
        # Decord returns RGB. The original local infer_videos.py converts to BGR
        # before passing frames through PTLFlow's IOAdapter, so keep that behavior.
        frames.append(vr[idx].asnumpy()[:, :, ::-1])
    return frames, native_fps


def _prepare_adapter(
    model: BaseModel,
    first_frame: np.ndarray,
    input_size: list[int],
    scale_factor: float | None,
    fp16: bool,
) -> IOAdapter:
    kwargs = {
        "output_stride": model.output_stride,
        "input_size": first_frame.shape[:2],
        "cuda": torch.cuda.is_available(),
        "fp16": fp16,
    }
    if scale_factor is not None:
        kwargs["target_scale_factor"] = scale_factor
    else:
        kwargs["target_size"] = input_size
    return IOAdapter(**kwargs)


def _tensor_dict_to_numpy_batched(tensor_dict: dict) -> dict:
    npy_dict = {}
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if len(value.shape) == 5 and key == "flows":
                value = value.permute(1, 0, 3, 4, 2).numpy()
            elif len(value.shape) == 4:
                value = value.permute(0, 2, 3, 1).numpy()
            elif len(value.shape) == 5 and key == "images":
                value = value.permute(0, 1, 3, 4, 2).numpy()
        npy_dict[key] = value
    return npy_dict


def _select_flow(preds_npy: dict, model_name: str) -> np.ndarray:
    flow = preds_npy["flows"]

    if flow.ndim == 5 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.ndim == 5 and flow.shape[1] == 1:
        flow = flow[:, 0]

    # MEMFOF consumes a 3-frame window and returns two flows for the center
    # frame: center->previous and center->next. For video visualization we want
    # the forward flow center->next.
    if model_name.lower() == "memfof" and flow.ndim == 4 and flow.shape[0] >= 2:
        flow = flow[1]

    if flow.ndim == 4 and flow.shape[0] == 1:
        flow = flow[0]

    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Unexpected flow shape for {model_name}: {flow.shape}")

    return flow


@torch.no_grad()
def infer(args: Namespace, model: BaseModel) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        if args.fp16:
            model = model.half()

    frames, native_fps = _frames_from_video(args.input_path, args.extract_fps)
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames in {args.input_path}")

    output_fps = float(args.output_fps) if args.output_fps and args.output_fps > 0 else native_fps
    io_adapter = _prepare_adapter(model, frames[0], args.input_size, args.scale_factor, args.fp16)

    flow_viz_frames = []
    raw_flows = []
    model_name = args.model.class_path.split(".")[-1]
    is_memfof = model_name.lower() == "memfof"

    for frame_idx in tqdm(range(0, len(frames) - 1), desc="PTLFlow frame pairs"):
        if is_memfof:
            prev_idx = max(frame_idx - 1, 0)
            inputs = io_adapter.prepare_inputs(
                [frames[prev_idx], frames[frame_idx], frames[frame_idx + 1]]
            )
        else:
            inputs = io_adapter.prepare_inputs([frames[frame_idx], frames[frame_idx + 1]])
        preds = model(inputs)
        preds["images"] = inputs["images"]
        preds = io_adapter.unscale(preds)
        preds_npy = _tensor_dict_to_numpy_batched(preds)
        flow = _select_flow(preds_npy, model_name)

        flow_viz_bgr = flow_to_rgb(flow)[:, :, ::-1]
        flow_viz_rgb = flow_viz_bgr[:, :, ::-1]
        flow_viz_frames.append(flow_viz_rgb)
        raw_flows.append(flow.astype(np.float16))

        if args.save_frames:
            stem = f"frame_{frame_idx:06d}"
            viz_dir = output_dir / "frames" / "flow_viz"
            flow_dir = output_dir / "frames" / "flow"
            viz_dir.mkdir(parents=True, exist_ok=True)
            flow_dir.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(viz_dir / f"{stem}.png"), flow_viz_bgr)
            flow_write(flow_dir / f"{stem}.{args.flow_format}", flow)

    video_path = output_dir / "flow_viz.mp4"
    with imageio.get_writer(video_path, fps=output_fps, codec="libx264", macro_block_size=None) as writer:
        for frame in flow_viz_frames:
            writer.append_data(frame)

    if args.save_raw_npz:
        np.savez_compressed(output_dir / "flow_raw.npz", flows=np.stack(raw_flows, axis=0))

    (output_dir / "metadata.txt").write_text(
        "\n".join(
            [
                f"input_path={args.input_path}",
                f"model={model_name}",
                f"ckpt_path={args.ckpt_path}",
                f"native_fps={native_fps}",
                f"extract_fps={args.extract_fps}",
                f"output_fps={output_fps}",
                f"num_input_frames={len(frames)}",
                f"num_flow_frames={len(flow_viz_frames)}",
            ]
        )
        + "\n"
    )


def main() -> None:
    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        parser_kwargs={"parents": [_base_parser()]},
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
    )
    cfg = cli.config
    model = ptlflow.restore_model(cli.model, cfg.ckpt_path)
    infer(cfg, model)


if __name__ == "__main__":
    main()
