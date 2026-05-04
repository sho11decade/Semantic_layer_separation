from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    segmenter = None
    if settings.sam2_checkpoint and settings.sam2_model_config:
        segmenter = SAM2Segmenter(checkpoint=settings.sam2_checkpoint, model_config=settings.sam2_model_config)
    else:
        segmenter = SimpleBoxSegmenter()

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