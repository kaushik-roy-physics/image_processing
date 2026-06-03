"""Shared helpers used by more than one pipeline step."""

from __future__ import annotations

import json
from pathlib import Path

from .image_loader import METADATA_NAME, result_dir_for


def load_image_metadata(out_dir: Path) -> dict:
    """Load ``image_metadata.json`` written by :class:`ImageLoader`."""
    meta_path = out_dir / METADATA_NAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{METADATA_NAME} missing in {out_dir}. Run the 'load' step first."
        )
    with open(meta_path) as f:
        return json.load(f)


def resolve_channel_path(out_dir: Path, ch_idx: int, mode: str, is_zstack: bool) -> Path:
    """
    Resolve the channel ``.tif`` appropriate for the segmentation/extraction mode.

    * ``3d`` → the per-channel z-stack ``ch{n}.tif`` (requires a z-stack input).
    * ``2d`` → the per-channel MIP ``ch{n}_MIP.tif`` when the input was a
      z-stack, otherwise the already-2-D ``ch{n}.tif``.
    """
    if mode == "3d":
        return out_dir / f"ch{ch_idx}.tif"
    if is_zstack:
        mip = out_dir / f"ch{ch_idx}_MIP.tif"
        if mip.exists():
            return mip
    return out_dir / f"ch{ch_idx}.tif"


__all__ = ["load_image_metadata", "resolve_channel_path", "result_dir_for"]
