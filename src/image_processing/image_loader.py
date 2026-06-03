"""
ImageLoader — step 1 of the pipeline.

For each input file it creates a sibling result directory named after the file
stem and writes:

* ``image_metadata.json``  — format-agnostic metadata + detected acquisition type
* ``multichannel.tif``     — the full image (``CYX`` for a MIP, ``CZYX`` for a stack)
* ``ch{n}.tif``            — each channel individually (2-D MIP or 3-D stack)

When the input is a **multichannel z-stack** it additionally writes the
maximum-intensity projections used by 2-D segmentation / extraction:

* ``multichannel_MIP.tif`` — ``CYX`` MIP across Z
* ``ch{n}_MIP.tif``        — per-channel MIP across Z

A ``.imgproc_output`` marker makes the directory invisible to the walker, and a
``.loaded`` checkpoint lets the step be skipped on re-runs unless forced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tifffile

from .directory_walker import OUTPUT_MARKER
from .io import read_image

logger = logging.getLogger(__name__)

CHECKPOINT = ".loaded"
METADATA_NAME = "image_metadata.json"


def result_dir_for(path: Path) -> Path:
    """Canonical per-image output directory: a folder named after the file stem."""
    return path.parent / path.stem


class ImageLoader:
    def __init__(self, cfg: dict):
        vis = cfg.get("visualization", {})
        self.dpi: int = int(vis.get("dpi", 300))

    def process(self, path: Path, force: bool = False) -> Path:
        out_dir = result_dir_for(path)
        checkpoint = out_dir / CHECKPOINT

        if checkpoint.exists() and not force:
            logger.debug("ImageLoader: %s already loaded.", path.name)
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / OUTPUT_MARKER).touch()
        logger.info("ImageLoader: reading %s", path.name)

        data, meta = read_image(path)
        n_ch = int(meta["n_channels"])
        is_zstack = bool(meta["is_zstack"])
        meta["n_channels_saved"] = n_ch

        with open(out_dir / METADATA_NAME, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        if is_zstack:
            # data: (C, Z, Y, X)
            _save(out_dir / "multichannel.tif", data, axes="CZYX")
            for ch in range(n_ch):
                _save(out_dir / f"ch{ch}.tif", data[ch], axes="ZYX")

            mip = data.max(axis=1)                       # (C, Y, X)
            _save(out_dir / "multichannel_MIP.tif", mip, axes="CYX")
            for ch in range(n_ch):
                _save(out_dir / f"ch{ch}_MIP.tif", mip[ch], axes="YX")

            shape_str = "×".join(str(s) for s in data.shape[1:])
            logger.info(
                "ImageLoader: %s — z-stack, %d ch, %s (+ MIPs)",
                path.name, n_ch, shape_str,
            )
        else:
            # data: (C, Y, X)
            _save(out_dir / "multichannel.tif", data, axes="CYX")
            for ch in range(n_ch):
                _save(out_dir / f"ch{ch}.tif", data[ch], axes="YX")

            shape_str = "×".join(str(s) for s in data.shape[1:])
            logger.info(
                "ImageLoader: %s — MIP, %d ch, %s", path.name, n_ch, shape_str,
            )

        checkpoint.touch()
        return out_dir


def _save(path: Path, arr: np.ndarray, axes: str) -> None:
    """Write a TIFF preserving the native dtype and recording the axis order."""
    tifffile.imwrite(
        path,
        np.ascontiguousarray(arr),
        photometric="minisblack",
        metadata={"axes": axes},
    )
