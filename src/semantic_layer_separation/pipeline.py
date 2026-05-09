from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from semantic_layer_separation.config import Settings
from semantic_layer_separation.errors import ConfigurationError, ProcessingError, ModelError, safe_execute, ErrorSeverity


@dataclass(slots=True)
class PipelineResult:
    targets: list[str]
    boxes: list[BoundingBox]
    output_dir: Path


def run_pipeline(*, image_path: Path, settings: Settings, prompt: str | None = None, output_subdir: str | None = None) -> PipelineResult:
    from semantic_layer_separation.detectors.grounding_dino import BoundingBox, GroundingDINODetector
    from semantic_layer_separation.exporters.image_export import ensure_output_dir, sanitize_label, save_cutout, save_mask, save_metadata, save_overlay
    from semantic_layer_separation.providers.azure_openai import AzureOpenAIPlanner
    from semantic_layer_separation.segmenters.sam2 import SAM2Segmenter, SegmentationMask, SimpleBoxSegmenter

    if output_subdir:
        output_dir = ensure_output_dir(Path(settings.output_dir) / output_subdir)
    else:
        output_dir = ensure_output_dir(settings.output_dir)

    try:
        planner = AzureOpenAIPlanner(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            deployment=settings.azure_openai_deployment,
        )
        planning = planner.plan(
            image_path=image_path,
            prompt=prompt,
            max_targets=settings.planning_max_targets,
        )
    except Exception as exc:
        raise ModelError(f"Azure OpenAI planning failed: {exc}") from exc

    try:
        detector = GroundingDINODetector(settings.grounding_dino_model)
        boxes = detector.detect(
            image_path,
            planning.targets,
            threshold=settings.detection_box_threshold,
            text_threshold=settings.detection_text_threshold,
            nms_iou_threshold=settings.detection_nms_iou_threshold,
            max_per_label=settings.detection_max_per_label,
        )
    except Exception as exc:
        raise ModelError(f"Grounding DINO detection failed: {exc}") from exc

    segmenter = _build_segmenter(settings, SAM2Segmenter, SimpleBoxSegmenter)

    try:
        masks = segmenter.segment(image_path, [(box.label, box.box) for box in boxes])
    except Exception as exc:
        raise ProcessingError(f"Segmentation failed: {exc}") from exc

    if settings.background_residual_enabled:
        masks = _append_residual_background_mask(
            image_path=image_path,
            masks=masks,
            min_area_ratio=settings.background_residual_min_area_ratio,
            label=settings.background_residual_label,
            segmentation_mask_cls=SegmentationMask,
        )
    
    layers_info = []
    for index, mask in tqdm(enumerate(masks, start=1), total=len(masks), desc="Processing layers"):
        clean_label = sanitize_label(mask.label)
        mask_path = output_dir / f"{index:02d}_{clean_label}_mask.png"
        cutout_path = output_dir / f"{index:02d}_{clean_label}_cutout.png"
        overlay_path = output_dir / f"{index:02d}_{clean_label}_overlay.png"
        
        save_mask(mask.mask, mask_path)
        save_cutout(image_path, mask.mask, cutout_path)
        save_overlay(image_path, mask.mask, overlay_path)
        
        layers_info.append({
            "index": index,
            "label": mask.label,
            "clean_label": clean_label,
            "mask_file": mask_path.name,
            "cutout_file": cutout_path.name,
            "overlay_file": overlay_path.name,
        })
    
    save_metadata(layers_info, output_dir)

    return PipelineResult(targets=planning.targets, boxes=boxes, output_dir=output_dir)


def _append_residual_background_mask(*, image_path: Path, masks: list, min_area_ratio: float, label: str, segmentation_mask_cls) -> list:
    with Image.open(image_path).convert("RGB") as source_image:
        width, height = source_image.size
    if width <= 0 or height <= 0:
        return masks

    uncovered = np.ones((height, width), dtype=bool)
    for mask_item in masks:
        mask_array = np.asarray(mask_item.mask, dtype=bool)
        if mask_array.shape != uncovered.shape:
            print(
                f"[semantic-layer-separation] Skipping mask with unexpected shape: {mask_array.shape} (expected {uncovered.shape})",
                file=sys.stderr,
            )
            continue
        uncovered &= ~mask_array

    residual_ratio = float(uncovered.sum() / uncovered.size)
    if residual_ratio < min_area_ratio:
        return masks

    residual_label = label.strip() or "background"
    return [*masks, segmentation_mask_cls(label=residual_label, mask=uncovered)]


def _build_segmenter(settings: Settings, sam2_segmenter_cls, simple_box_segmenter_cls):
    if not settings.sam2_checkpoint or not settings.sam2_model_config:
        return simple_box_segmenter_cls()

    try:
        return sam2_segmenter_cls(checkpoint=settings.sam2_checkpoint, model_config=settings.sam2_model_config)
    except Exception as exc:  # pragma: no cover - runtime fallback for optional SAM2 setup
        print(
            f"[semantic-layer-separation] SAM 2 initialization failed, falling back to rectangular masks: {exc}",
            file=sys.stderr,
        )
        return simple_box_segmenter_cls()


def run_batch_pipeline(*, image_dir: Path, settings: Settings, prompt: str | None = None) -> list[dict]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ConfigurationError(f"Image directory not found: {image_dir}")
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        raise ConfigurationError(f"No image files found in {image_dir}")
    
    results = []
    for image_path in tqdm(image_files, desc="Batch processing"):
        subdir_name = image_path.stem
        try:
            result = run_pipeline(image_path=image_path, settings=settings, prompt=prompt, output_subdir=subdir_name)
            results.append({
                "image": image_path,
                "output_dir": result.output_dir,
                "targets": result.targets,
                "boxes": result.boxes,
            })
        except Exception as exc:
            print(
                f"[semantic-layer-separation] Failed to process {image_path.name}: {exc}",
                file=sys.stderr,
            )
            results.append({
                "image": image_path,
                "error": str(exc),
                "targets": [],
                "boxes": [],
            })
    
    return results
