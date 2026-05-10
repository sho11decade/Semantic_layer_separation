from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

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

    if settings.drawing_completion_enabled:
        masks = _append_drawing_process_completion_masks(
            image_path=image_path,
            masks=masks,
            min_area_ratio=settings.drawing_completion_min_area_ratio,
            shadow_luma_threshold=settings.drawing_completion_shadow_luma_threshold,
            edge_quantile=settings.drawing_completion_edge_quantile,
            include_base=settings.drawing_completion_base_enabled,
            include_shadow=settings.drawing_completion_shadow_enabled,
            include_line=settings.drawing_completion_line_enabled,
            segmentation_mask_cls=SegmentationMask,
        )

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


def _append_drawing_process_completion_masks(
    *,
    image_path: Path,
    masks: list,
    min_area_ratio: float,
    shadow_luma_threshold: float,
    edge_quantile: float,
    include_base: bool,
    include_shadow: bool,
    include_line: bool,
    segmentation_mask_cls,
) -> list:
    if not masks:
        return masks

    with Image.open(image_path).convert("RGB") as source_image:
        image_array = np.asarray(source_image, dtype=np.float32)

    height, width = image_array.shape[:2]
    if width <= 0 or height <= 0:
        return masks

    covered = np.zeros((height, width), dtype=bool)
    for mask_item in masks:
        mask_array = np.asarray(mask_item.mask, dtype=bool)
        if mask_array.shape != covered.shape:
            print(
                f"[semantic-layer-separation] Skipping mask with unexpected shape for drawing completion: "
                f"{mask_array.shape} (expected {covered.shape})",
                file=sys.stderr,
            )
            continue
        covered |= mask_array

    covered_ratio = float(covered.sum() / covered.size)
    if covered_ratio < min_area_ratio:
        return masks

    normalized_labels = [_normalize_label(mask_item.label) for mask_item in masks]
    additions: list = []

    if include_base and not _has_any_keyword(normalized_labels, ("base", "flat", "underpaint", "下塗り", "ベース")):
        additions.append(segmentation_mask_cls(label="base_fill", mask=covered.copy()))

    gray = (
        (0.299 * image_array[..., 0] + 0.587 * image_array[..., 1] + 0.114 * image_array[..., 2]) / 255.0
    ).astype(np.float32)

    if include_shadow and not _has_any_keyword(normalized_labels, ("shadow", "shade", "影")):
        shadow_mask = covered & (gray <= shadow_luma_threshold)
        if float(shadow_mask.sum() / shadow_mask.size) >= min_area_ratio:
            additions.append(segmentation_mask_cls(label="shadow", mask=shadow_mask))

    if include_line and not _has_any_keyword(normalized_labels, ("line", "line art", "lineart", "outline", "ink", "線画")):
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[0:1, :]))
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, 0:1]))
        gradient = gx + gy
        covered_gradients = gradient[covered]
        if covered_gradients.size > 0:
            threshold = float(np.quantile(covered_gradients, edge_quantile))
            line_mask = covered & (gradient >= threshold)
            if float(line_mask.sum() / line_mask.size) >= min_area_ratio:
                additions.append(segmentation_mask_cls(label="line_art", mask=line_mask))

    return [*masks, *additions]


def _normalize_label(label: str) -> str:
    return " ".join(str(label).replace("_", " ").lower().split())


def _has_any_keyword(normalized_labels: list[str], keywords: tuple[str, ...]) -> bool:
    return any(any(keyword in normalized for keyword in keywords) for normalized in normalized_labels)


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


def _count_exported_layers(output_dir: Path) -> int:
    metadata_path = output_dir / "layers.json"
    if not metadata_path.exists():
        return 0
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    layers = payload.get("layers", [])
    return len(layers) if isinstance(layers, list) else 0


def run_benchmark_pipeline(
    *,
    image_dir: Path,
    settings: Settings,
    prompt: str | None = None,
    output_report_path: Path | None = None,
) -> dict:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ConfigurationError(f"Image directory not found: {image_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = sorted([f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])

    if not image_files:
        raise ConfigurationError(f"No image files found in {image_dir}")

    benchmark_started = perf_counter()
    results: list[dict] = []

    for image_path in tqdm(image_files, desc="Benchmark processing"):
        started = perf_counter()
        subdir_name = image_path.stem
        try:
            result = run_pipeline(image_path=image_path, settings=settings, prompt=prompt, output_subdir=subdir_name)
            elapsed_ms = (perf_counter() - started) * 1000.0
            results.append(
                {
                    "image": str(image_path),
                    "status": "ok",
                    "duration_ms": round(elapsed_ms, 2),
                    "target_count": len(result.targets),
                    "box_count": len(result.boxes),
                    "layer_count": _count_exported_layers(result.output_dir),
                    "output_dir": str(result.output_dir),
                }
            )
        except Exception as exc:
            elapsed_ms = (perf_counter() - started) * 1000.0
            print(
                f"[semantic-layer-separation] Benchmark failed for {image_path.name}: {exc}",
                file=sys.stderr,
            )
            results.append(
                {
                    "image": str(image_path),
                    "status": "error",
                    "duration_ms": round(elapsed_ms, 2),
                    "target_count": 0,
                    "box_count": 0,
                    "layer_count": 0,
                    "error": str(exc),
                }
            )

    total_duration_ms = (perf_counter() - benchmark_started) * 1000.0
    success_results = [item for item in results if item["status"] == "ok"]
    error_results = [item for item in results if item["status"] == "error"]
    avg_duration_ms = round(
        sum(item["duration_ms"] for item in success_results) / len(success_results), 2
    ) if success_results else None

    report = {
        "mode": "benchmark",
        "image_dir": str(image_dir),
        "summary": {
            "total_images": len(results),
            "successful_images": len(success_results),
            "failed_images": len(error_results),
            "total_duration_ms": round(total_duration_ms, 2),
            "avg_duration_ms_success_only": avg_duration_ms,
        },
        "results": results,
    }

    if output_report_path is not None:
        report_path = Path(output_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)

    return report
