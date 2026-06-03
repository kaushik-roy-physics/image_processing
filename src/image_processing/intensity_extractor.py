"""
IntensityExtractor — step 3 of the pipeline.

For every marker channel listed in ``channels.marker_channels`` it maps the
channel image onto the nuclear masks and computes a per-nucleus median
intensity (robust to hot pixels). Results are written to
``per_nuclei_intensities.joblib`` containing both **raw** and **DAPI-normalised**
intensities, the per-nucleus DAPI level, mask sizes, and **nucleus centroids**
(``(y, x)`` for 2-D, ``(z, y, x)`` for 3-D) for spatial QC of outliers.

All per-label statistics are vectorised: intensities via ``scipy.ndimage``,
centroids and sizes via coordinate ``bincount`` — no per-nucleus Python loop.
Works identically for 2-D masks (MIP) and 3-D masks (z-stack).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import tifffile
from scipy.ndimage import median as nd_median

from .utils import load_image_metadata, resolve_channel_path

logger = logging.getLogger(__name__)

CHECKPOINT = ".extracted"
OUTPUT_NAME = "per_nuclei_intensities.joblib"


class IntensityExtractor:
    def __init__(self, cfg: dict):
        chan = cfg.get("channels", {})
        self.dapi_index: int = int(chan.get("dapi_index", 0))
        self.marker_channels: list[int] = [int(c) for c in chan.get("marker_channels", [])]
        self.normalize_by_dapi: bool = bool(
            cfg.get("intensity", {}).get("normalize_by_dapi", True)
        )
        # Mode follows segmentation so masks and channels stay consistent.
        self.mode: str = (cfg.get("segmentation", {}).get("mode") or "2d").lower()

    def process(self, path: Path, force: bool = False) -> None:
        from .image_loader import result_dir_for
        out_dir = result_dir_for(path)
        checkpoint = out_dir / CHECKPOINT
        if checkpoint.exists() and not force:
            logger.debug("IntensityExtractor: %s already extracted.", path.name)
            return

        meta = load_image_metadata(out_dir)
        is_zstack = bool(meta.get("is_zstack", False))
        channel_names = meta.get("channel_names")

        masks_path = out_dir / "nuclear_masks.tif"
        if not masks_path.exists():
            raise FileNotFoundError(
                f"Masks missing at {masks_path}. Run the 'segment' step first."
            )
        masks = tifffile.imread(masks_path)
        labels = np.unique(masks)
        labels = labels[labels > 0]

        result: dict = {
            "n_cells": int(labels.size),
            "labels": labels.astype(np.int64),
            "mode": self.mode,
            "is_zstack": is_zstack,
            "dapi_index": self.dapi_index,
            "marker_channels": list(self.marker_channels),
            "centroid_axes": "zyx" if masks.ndim == 3 else "yx",
            "raw_intensities": {},
            "dapi_normalized_intensities": {},
            "global_stats": {},
            "channel_names": {},
        }

        if labels.size == 0:
            logger.warning("IntensityExtractor: no nuclei in %s", path.name)
            joblib.dump(result, out_dir / OUTPUT_NAME)
            checkpoint.touch()
            return

        # Centroids + mask sizes (vectorised over all labels at once).
        result["centroids"], result["mask_sizes"] = _centroids_and_sizes(masks, labels)

        # DAPI per nucleus (for normalisation and QC).
        dapi_per_cell = None
        dapi_path = resolve_channel_path(out_dir, self.dapi_index, self.mode, is_zstack)
        if dapi_path.exists():
            dapi_img = tifffile.imread(dapi_path).astype(np.float32)
            dapi_per_cell = nd_median(dapi_img, labels=masks, index=labels).astype(np.float32)
        else:
            logger.warning("IntensityExtractor: DAPI channel %s missing — "
                           "normalisation disabled for %s", dapi_path.name, path.name)
        result["dapi_per_cell"] = dapi_per_cell
        result["channel_names"][self.dapi_index] = _name_for(channel_names, self.dapi_index, "DAPI")

        for ch in self.marker_channels:
            ch_path = resolve_channel_path(out_dir, ch, self.mode, is_zstack)
            if not ch_path.exists():
                logger.warning("IntensityExtractor: channel file %s missing for %s",
                               ch_path.name, path.name)
                continue
            img = tifffile.imread(ch_path).astype(np.float32)
            per_cell = nd_median(img, labels=masks, index=labels).astype(np.float32)

            result["raw_intensities"][ch] = per_cell
            result["global_stats"][ch] = _stats(per_cell)
            result["channel_names"][ch] = _name_for(channel_names, ch, f"ch{ch}")

            if self.normalize_by_dapi and dapi_per_cell is not None:
                norm = np.divide(per_cell, dapi_per_cell,
                                 out=np.zeros_like(per_cell),
                                 where=dapi_per_cell > 0).astype(np.float32)
                result["dapi_normalized_intensities"][ch] = norm

        joblib.dump(result, out_dir / OUTPUT_NAME)
        checkpoint.touch()
        logger.info(
            "IntensityExtractor [%s]: %s — %d nuclei, channels %s",
            self.mode, path.name, labels.size, self.marker_channels,
        )


# ---------------------------------------------------------------------------
# Vectorised geometry
# ---------------------------------------------------------------------------

def _centroids_and_sizes(masks: np.ndarray, labels: np.ndarray):
    """
    Geometric centroids and voxel/pixel counts for every label, computed from a
    single ``nonzero`` pass. Returns ``(centroids (n, ndim), sizes (n,))`` with
    rows ordered to match ``labels``.
    """
    coords = np.nonzero(masks)               # tuple of (z,)y,x index arrays
    flat_labels = masks[coords]
    n_max = int(masks.max())

    counts = np.bincount(flat_labels, minlength=n_max + 1).astype(np.float64)
    sums = [np.bincount(flat_labels, weights=c.astype(np.float64), minlength=n_max + 1)
            for c in coords]

    sel = labels                              # 1-based label ids
    safe_counts = np.where(counts[sel] > 0, counts[sel], 1.0)
    centroids = np.stack([s[sel] / safe_counts for s in sums], axis=1).astype(np.float32)
    sizes = counts[sel].astype(np.int64)
    return centroids, sizes


def _name_for(channel_names, idx: int, default: str) -> str:
    if channel_names and 0 <= idx < len(channel_names):
        return str(channel_names[idx])
    return default


def _stats(arr: np.ndarray) -> dict:
    return {
        "n_cells": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }
