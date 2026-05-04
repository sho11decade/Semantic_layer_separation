from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic layer separation MVP")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--prompt", default=None, help="Optional custom planning prompt")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from semantic_layer_separation.config import load_settings
    from semantic_layer_separation.pipeline import run_pipeline

    settings = load_settings()
    result = run_pipeline(image_path=args.image, settings=settings, prompt=args.prompt)

    print(
        json.dumps(
            {
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())