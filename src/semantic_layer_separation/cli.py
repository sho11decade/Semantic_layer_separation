from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic layer separation MVP")
    
    # Add validation mode option
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration and exit")
    
    # Create mutually exclusive group for processing modes
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--image", type=Path, help="Input image path (single image)")
    group.add_argument("--image-dir", type=Path, help="Input directory with images (batch processing)")
    group.add_argument("--benchmark-dir", type=Path, help="Input directory with images (benchmark mode)")
    parser.add_argument("--benchmark-report", type=Path, default=None, help="Path to benchmark JSON report")
    
    parser.add_argument("--prompt", default=None, help="Optional custom planning prompt")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Setup logging
    from semantic_layer_separation.logging_config import setup_logging, get_logger
    setup_logging(log_level=getattr(logging, args.log_level))
    logger = get_logger(__name__)
    
    logger.info(f"Starting semantic layer separation (log_level={args.log_level})")

    from semantic_layer_separation.config import load_settings
    from semantic_layer_separation.validators import print_validation_report
    
    settings = load_settings()
    logger.debug(f"Settings loaded from .env")
    
    # Handle validation mode
    if args.validate_config:
        logger.info("Running configuration validation")
        is_valid = print_validation_report(settings)
        return 0 if is_valid else 1
    
    # Check that one processing mode is provided
    if not args.image and not args.image_dir and not args.benchmark_dir:
        logger.error("Either --image, --image-dir or --benchmark-dir must be specified")
        print("Error: Either --image, --image-dir or --benchmark-dir must be specified")
        return 1
    
    from semantic_layer_separation.pipeline import run_pipeline, run_batch_pipeline, run_benchmark_pipeline
    
    if args.image:
        # Single image mode
        logger.info(f"Processing single image: {args.image}")
        result = run_pipeline(image_path=args.image, settings=settings, prompt=args.prompt)
        logger.info(f"Single image processing completed: {len(result.targets)} targets, {len(result.boxes)} boxes")
        print(
            json.dumps(
                {
                    "mode": "single",
                    "targets": result.targets,
                    "boxes": [
                        {
                            "label": box.label,
                            "score": box.score,
                            "box": list(box.box),
                        }
                        for box in result.boxes
                    ],
                    "output_dir": str(result.output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.image_dir:
        # Batch mode
        logger.info(f"Processing batch from directory: {args.image_dir}")
        results = run_batch_pipeline(image_dir=args.image_dir, settings=settings, prompt=args.prompt)
        logger.info(f"Batch processing completed: {len(results)} images processed")
        print(
            json.dumps(
                {
                    "mode": "batch",
                    "processed_images": len(results),
                    "results": [
                        {
                            "image": str(r["image"]),
                            "output_dir": str(r.get("output_dir", "")),
                            "target_count": len(r["targets"]),
                            "box_count": len(r["boxes"]),
                            "error": r.get("error"),
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        # Benchmark mode
        logger.info(f"Running benchmark for directory: {args.benchmark_dir}")
        report_path = args.benchmark_report or (Path(settings.output_dir) / "benchmark_report.json")
        report = run_benchmark_pipeline(
            image_dir=args.benchmark_dir,
            settings=settings,
            prompt=args.prompt,
            output_report_path=report_path,
        )
        logger.info(
            "Benchmark completed: "
            f"{report['summary']['successful_images']} success / {report['summary']['total_images']} total"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
