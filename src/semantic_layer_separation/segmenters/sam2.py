from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import os
import re

import numpy as np
from PIL import Image

try:
    from sam2.build_sam import build_sam2
    from sam2.build_sam import _load_checkpoint
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional dependency
    build_sam2 = None
    _load_checkpoint = None
    SAM2ImagePredictor = None
    compose = None
    initialize_config_dir = None
    instantiate = None
    OmegaConf = None


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

        checkpoint_path = self._resolve_checkpoint_path(checkpoint)
        model = self._build_model(model_config, checkpoint_path)
        self._predictor = SAM2ImagePredictor(model)

    @staticmethod
    def _resolve_checkpoint_path(checkpoint: str) -> str:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    @staticmethod
    def _split_config_path(config_path: Path) -> tuple[Path, str]:
        config_path = config_path.expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"SAM2 config not found: {config_path}")

        config_root = None
        for parent in config_path.parents:
            if parent.name in {"configs", "conf"}:
                config_root = parent
                break

        if config_root is None:
            config_root = config_path.parent

        config_name = config_path.relative_to(config_root).as_posix()
        return config_root, config_name

    @classmethod
    def _build_model(cls, model_config: str, checkpoint_path: str):
        config_path = Path(model_config).expanduser()
        if config_path.suffix in {".yaml", ".yml"} or config_path.exists():
            config_dir, config_name = cls._split_config_path(config_path)
            if initialize_config_dir is None or compose is None or instantiate is None or OmegaConf is None or _load_checkpoint is None:
                raise RuntimeError("SAM2 Hydra dependencies are not available")

            with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                cfg = compose(
                    config_name=config_name,
                    overrides=[
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                    ],
                )
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)
            _load_checkpoint(model, checkpoint_path)
            return model.to("cpu")

        return build_sam2(model_config, checkpoint_path, device="cpu")

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