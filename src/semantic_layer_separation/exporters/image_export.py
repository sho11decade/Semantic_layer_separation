from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_label(label: str) -> str:
    """Sanitize label for use as a filename."""
    label = label.strip()
    label = re.sub(r'\s+', '_', label)
    label = re.sub(r'[^a-zA-Z0-9_\-]', '', label)
    label = re.sub(r'_+', '_', label)
    label = label.strip('_')
    return label or "layer"


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


def save_metadata(layers_info: list[dict], output_dir: Path, relations: dict | None = None) -> None:
    """Save layer metadata to JSON."""
    metadata = {
        "layers": layers_info,
        "version": "1.1",
    }
    if relations is not None:
        metadata["relations"] = relations
    with open(output_dir / "layers.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
