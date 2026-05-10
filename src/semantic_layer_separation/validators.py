"""Configuration validation utility."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch


def validate_config(settings) -> Tuple[bool, list[str]]:
    """Validate all configuration settings.
    
    Args:
        settings: Settings object from config module
        
    Returns:
        Tuple of (is_valid: bool, messages: list[str])
    """
    messages = []
    
    # Check Azure OpenAI settings
    if not settings.azure_openai_api_key:
        messages.append("❌ Azure OpenAI API key not set")
    else:
        messages.append("✅ Azure OpenAI API key found")
    
    if not settings.azure_openai_endpoint:
        messages.append("❌ Azure OpenAI endpoint not set")
    else:
        messages.append(f"✅ Azure OpenAI endpoint: {settings.azure_openai_endpoint}")
    
    if not settings.azure_openai_deployment:
        messages.append("❌ Azure OpenAI deployment name not set")
    else:
        messages.append(f"✅ Azure OpenAI deployment: {settings.azure_openai_deployment}")
    
    # Check Grounding DINO model
    if not settings.grounding_dino_model:
        messages.append("❌ Grounding DINO model not specified")
    else:
        messages.append(f"✅ Grounding DINO model: {settings.grounding_dino_model}")
        messages.append(
            "✅ Detection params: "
            f"box={settings.detection_box_threshold}, "
            f"text={settings.detection_text_threshold}, "
            f"nms_iou={settings.detection_nms_iou_threshold}, "
            f"max_per_label={settings.detection_max_per_label}"
        )
        messages.append(f"✅ Planning max targets: {settings.planning_max_targets}")
        messages.append(
            "✅ Background residual: "
            f"enabled={settings.background_residual_enabled}, "
            f"min_area_ratio={settings.background_residual_min_area_ratio}, "
            f"label={settings.background_residual_label}"
        )
        messages.append(
            "✅ Drawing completion: "
            f"enabled={settings.drawing_completion_enabled}, "
            f"base={settings.drawing_completion_base_enabled}, "
            f"shadow={settings.drawing_completion_shadow_enabled}, "
            f"line={settings.drawing_completion_line_enabled}, "
            f"min_area_ratio={settings.drawing_completion_min_area_ratio}, "
            f"shadow_luma_threshold={settings.drawing_completion_shadow_luma_threshold}, "
            f"edge_quantile={settings.drawing_completion_edge_quantile}"
        )
        try:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection
            messages.append("  → Loading model (this may take a while)...")
            _ = AutoProcessor.from_pretrained(settings.grounding_dino_model)
            _ = GroundingDinoForObjectDetection.from_pretrained(settings.grounding_dino_model)
            messages.append("  ✅ Grounding DINO model loads successfully")
        except Exception as e:
            messages.append(f"  ❌ Failed to load Grounding DINO model: {e}")
    
    # Check SAM2 settings (optional)
    if settings.sam2_checkpoint or settings.sam2_model_config:
        checkpoint_path = Path(settings.sam2_checkpoint).expanduser() if settings.sam2_checkpoint else None
        config_path = Path(settings.sam2_model_config).expanduser() if settings.sam2_model_config else None
        
        if checkpoint_path and not checkpoint_path.exists():
            messages.append(f"❌ SAM2 checkpoint not found: {checkpoint_path}")
        elif checkpoint_path:
            messages.append(f"✅ SAM2 checkpoint found: {checkpoint_path}")
        
        if config_path and not config_path.exists():
            messages.append(f"❌ SAM2 config not found: {config_path}")
        elif config_path:
            messages.append(f"✅ SAM2 config found: {config_path}")
        
        if checkpoint_path and config_path:
            try:
                from semantic_layer_separation.segmenters.sam2 import SAM2Segmenter
                _ = SAM2Segmenter(checkpoint=str(checkpoint_path), model_config=str(config_path))
                messages.append("✅ SAM2 model initializes successfully")
            except Exception as e:
                messages.append(f"⚠️  SAM2 initialization failed (will use fallback): {e}")
    else:
        messages.append("ℹ️  SAM2 not configured (will use rectangular fallback)")
    
    # Check output directory
    output_dir = Path(settings.output_dir).expanduser()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".validation_test"
        test_file.write_text("test")
        test_file.unlink()
        messages.append(f"✅ Output directory writable: {output_dir}")
    except Exception as e:
        messages.append(f"❌ Output directory not writable: {output_dir} ({e})")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        messages.append(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        messages.append("ℹ️  CUDA not available (will use CPU)")
    
    is_valid = not any(msg.startswith("❌") for msg in messages)
    return is_valid, messages


def print_validation_report(settings):
    """Print a human-readable validation report."""
    is_valid, messages = validate_config(settings)
    
    print("\n" + "="*60)
    print("Configuration Validation Report")
    print("="*60)
    
    for msg in messages:
        print(msg)
    
    print("="*60)
    if is_valid:
        print("✅ All critical settings are valid!")
    else:
        print("❌ Some critical settings are missing or invalid.")
        print("   Please check your .env file.")
    print("="*60 + "\n")
    
    return is_valid
