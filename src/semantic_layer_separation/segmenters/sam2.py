from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:  # pragma: no cover - optional dependency
    build_sam2 = None
    SAM2ImagePredictor = None


@dataclass(slots=True)
class SegmentationMask:
    label: str
    mask: np.ndarray


class SAM2Segmenter:
    def __init__(self, *, checkpoint: str | None, model_config: str | None) -> None:
        if build_sam2 is None or SAM2ImagePredictor is None:
            raise RuntimeError("sam2 package is not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        if not checkpoint or not model_config:
            raise ValueError("SAM2 checkpoint and model config are required")

        model = build_sam2(model_config, checkpoint, device="cpu")
        self._predictor = SAM2ImagePredictor(model)

    def segment(self, image_path: Path, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> list[SegmentationMask]:
        image = np.array(Image.open(image_path).convert("RGB"))
        self._predictor.set_image(image)

        results: list[SegmentationMask] = []
        for label, box in boxes:
            masks, _, _ = self._predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
            results.append(SegmentationMask(label=label, mask=masks[0].astype(bool)))
        return results


class SimpleBoxSegmenter:
    """Fallback segmenter that uses bounding boxes as rectangular masks."""

    def segment(self, image_path: Path, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> list[SegmentationMask]:
        image = Image.open(image_path).convert("RGB")
        height, width = image.size[::-1]
        results: list[SegmentationMask] = []

        for label, box in boxes:
            x0, y0, x1, y1 = box
            mask = np.zeros((height, width), dtype=bool)
            mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
            results.append(SegmentationMask(label=label, mask=mask))
        return results