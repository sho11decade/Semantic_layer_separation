from __future__ import annotations

from pathlib import Path

from PIL import Image

from semantic_layer_separation.viewer.loader import LayerAsset, LayerSet


def _safe_open_rgba(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("RGBA")


def _with_opacity(image: Image.Image, opacity: float) -> Image.Image:
    alpha = image.getchannel("A").point(lambda value: int(max(0, min(255, value * opacity))))
    output = image.copy()
    output.putalpha(alpha)
    return output


def build_composite_image(layer_set: LayerSet, visible_indices: set[int], opacity: float) -> Image.Image | None:
    base: Image.Image | None = None
    for layer in sorted(layer_set.layers, key=lambda item: item.index):
        if layer.index not in visible_indices:
            continue
        cutout = _safe_open_rgba(layer.cutout_path)
        if cutout is None:
            continue
        rendered = _with_opacity(cutout, opacity)
        if base is None:
            base = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
        if base.size != rendered.size:
            rendered = rendered.resize(base.size, Image.Resampling.BILINEAR)
        base = Image.alpha_composite(base, rendered)
    return base


def resolve_active_layer(active_index: int | None, layers: list[LayerAsset]) -> LayerAsset | None:
    if active_index is None:
        return None
    for layer in layers:
        if layer.index == active_index:
            return layer
    return None
