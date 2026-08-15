#!/usr/bin/env python3
"""
Standalone inference script — KLA Phase 1 requirement.

- Accepts an input-directory and output-directory argument (no other
  required arguments; everything else has a sensible default).
- Loads every degraded image in the input directory, restores it with the
  two-stage (DnCNN -> ESPCN) pipeline, and saves each output to the output
  directory, preserving filenames.
- Supports batched GPU execution.
- Does NOT require editing this file, notebook cells, or local paths —
  weights are loaded from --weights-dir (default: weights/).

Usage:
    python inference.py --input-dir path/to/NoisyLR --output-dir path/to/restored
    python inference.py --input-dir data/test --output-dir results/test_restored --batch-size 16

Timing note: end-to-end runtime printed at the end includes disk reads,
preprocessing, CPU->GPU transfer, model execution, GPU->CPU transfer,
post-processing and saving — per KLA's runtime definition.
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pipeline import TwoStageRestorer

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: str):
    return sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS])


def load_image(path: Path) -> np.ndarray:
    """Loads as float32 HWC RGB. Deliberately NOT clipped to [0,1] on load —
    KLA notes NoisyLR values may legitimately extend slightly outside that
    range, and clipping is applied inside the model/pipeline, not blindly
    at I/O time."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def save_image(path: Path, img: np.ndarray):
    """Clips to [0,1] and writes 8-bit PNG/JPEG. KLA scores the image
    exactly as saved, so clipping/normalization happens here, once,
    deliberately."""
    img = np.clip(img, 0.0, 1.0)
    img_uint8 = (img * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def to_batch_tensor(images: list) -> torch.Tensor:
    """List of HWC float32 arrays (possibly different sizes) -> a single
    batch tensor. Since inputs in a batch can vary slightly in size, this
    pads to the max H/W in the batch and crops back per-image after
    inference (see `restore_batch`)."""
    max_h = max(im.shape[0] for im in images)
    max_w = max(im.shape[1] for im in images)
    batch = np.zeros((len(images), max_h, max_w, 3), dtype=np.float32)
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        batch[i, :h, :w] = im
    tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())
    return tensor


def restore_batch(model: TwoStageRestorer, images: list, device: torch.device, upscale: int) -> list:
    shapes = [im.shape[:2] for im in images]
    batch = to_batch_tensor(images).to(device)
    with torch.no_grad():
        restored = model.restore(batch)
    restored = restored.cpu().numpy().transpose(0, 2, 3, 1)

    outputs = []
    for i, (h, w) in enumerate(shapes):
        out_h, out_w = h * upscale, w * upscale
        outputs.append(restored[i, :out_h, :out_w])
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Two-stage restoration inference")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of degraded (NoisyLR) images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write restored images to")
    parser.add_argument("--weights-dir", type=str, default="weights", help="Directory containing dncnn_best.pth and espcn_best.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--upscale-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = TwoStageRestorer(upscale_factor=args.upscale_factor).to(device)
    dncnn_path = os.path.join(args.weights_dir, "dncnn_best.pth")
    espcn_path = os.path.join(args.weights_dir, "espcn_best.pth")
    model.load_stage_weights(dncnn_path, espcn_path, map_location=device)
    model.eval()

    paths = list_images(args.input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")
    print(f"Found {len(paths)} images in {args.input_dir}")

    t_start = time.time()
    n_done = 0
    for i in range(0, len(paths), args.batch_size):
        batch_paths = paths[i: i + args.batch_size]
        images = [load_image(p) for p in batch_paths]
        restored = restore_batch(model, images, device, args.upscale_factor)
        for p, out_img in zip(batch_paths, restored):
            save_image(Path(args.output_dir) / p.name, out_img)
        n_done += len(batch_paths)
        print(f"  restored {n_done}/{len(paths)}")

    elapsed = time.time() - t_start
    print(f"Done. {len(paths)} images restored in {elapsed:.2f}s "
          f"({elapsed / len(paths) * 1000:.1f} ms/image, end-to-end).")


if __name__ == "__main__":
    main()
