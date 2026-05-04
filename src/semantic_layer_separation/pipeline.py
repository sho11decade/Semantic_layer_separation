from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from semantic_layer_separation.config import Settings


@dataclass(slots=True)
class PipelineResult:
    targets: list[str]
    boxes: list[BoundingBox]
    output_dir: Path


def run_pipeline(*, image_path: Path, settings: Settings, prompt: str | None = None) -> PipelineResult:
    from semantic_layer_separation.detectors.grounding_dino import BoundingBox, GroundingDINODetector
    from semantic_layer_separation.exporters.image_export import ensure_output_dir, sanitize_label, save_cutout, save_mask, save_metadata, save_overlay
    from semantic_layer_separation.providers.azure_openai import AzureOpenAIPlanner
    from semantic_layer_separation.segmenters.sam2 import SAM2Segmenter, SimpleBoxSegmenter

    output_dir = ensure_output_dir(settings.output_dir)

    planner = AzureOpenAIPlanner(
        api_key=settings.azure_openai_api_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        deployment=settings.azure_openai_deployment,
    )
    planning = planner.plan(image_path=image_path, prompt=prompt)

    detector = GroundingDINODetector(settings.grounding_dino_model)
    boxes = detector.detect(image_path, planning.targets)

    segmenter = _build_segmenter(settings, SAM2Segmenter, SimpleBoxSegmenter)

    masks = segmenter.segment(image_path, [(box.label, box.box) for box in boxes])
    
    layers_info = []
    for index, mask in enumerate(masks, start=1):
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