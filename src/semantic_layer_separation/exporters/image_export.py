from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image.save(output_path)


def save_cutout(image_path: Path, mask: np.ndarray, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.putalpha(alpha)
    image.save(output_path)


def save_overlay(image_path: Path, mask: np.ndarray, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = np.array(image)
    highlight = np.zeros_like(overlay)
    highlight[..., 0] = 255
    highlight[..., 3] = (mask.astype(np.uint8) * 100)
    blended = np.clip(overlay.astype(np.int16) + highlight.astype(np.int16), 0, 255).astype(np.uint8)
    Image.fromarray(blended, mode="RGBA").save(output_path)