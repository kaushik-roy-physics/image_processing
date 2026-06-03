"""
Segmentation — step 2 of the pipeline (Cellpose-SAM nuclear segmentation).

Mode selection (config ``segmentation.mode``)
---------------------------------------------
* ``2d``   — segment the DAPI MIP (``ch{dapi}_MIP.tif`` for stacks, else
  ``ch{dapi}.tif``). Works for both MIP and z-stack inputs.
* ``3d``   — segment the DAPI z-stack (``ch{dapi}.tif``) with anisotropy taken
  from the image metadata (``pixel_size_z / pixel_size_x``), falling back to
  2.0. Requesting 3-D on a MIP input is an error.
* unset    — defaults to ``2d`` on the DAPI MIP.

Optional cytoplasm channel
--------------------------
If ``channels.has_cyto`` and ``segmentation.use_cyto`` are both true, the DAPI
and cytoplasm channels are stacked and passed together to Cellpose-SAM. The
v4 model is channel-order agnostic and needs no explicit channel arguments.

Outputs (per image directory)
-----------------------------
* ``nuclear_masks.tif``       — label image (2-D, or a z-stack for 3-D mode)
* ``segmentation_overlay.png``— DAPI with nucleus outlines (extreme DAPI
  percentiles clipped); for 3-D also a ``..._montage.png`` across Z
* ``size_filter_qc.png``      — adaptive size/DAPI filter diagnostic
* ``segmentation_metadata.json`` — parameters + nucleus counts before/after
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from .utils import load_image_metadata, resolve_channel_path  # noqa: E402

logger = logging.getLogger(__name__)

CHECKPOINT = ".segmented"
_DEFAULT_ANISOTROPY = 2.0
_MIN_MASKS_FOR_ADAPTIVE = 10   # too few masks to fit a distribution → skip adaptive


def _check_gpu() -> bool:
    try:
        import torch
    except ImportError:
        logger.info("Segmentation: torch unavailable — Cellpose will use CPU")
        return False
    if not torch.cuda.is_available():
        logger.info("Segmentation: no GPU found — using CPU")
        return False
    try:
        logger.info("Segmentation: GPU — %s", torch.cuda.get_device_name(0))
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("Segmentation: GPU check failed (%s) — using CPU", e)
        return False


class Segmentation:
    def __init__(self, cfg: dict):
        seg = cfg.get("segmentation", {})
        chan = cfg.get("channels", {})

        mode = seg.get("mode")
        self.mode: str = (mode or "2d").lower()
        if self.mode not in ("2d", "3d"):
            raise ValueError(f"segmentation.mode must be '2d', '3d' or null (got {mode!r}).")

        self.dapi_index: int = int(chan.get("dapi_index", 0))
        self.has_cyto: bool = bool(chan.get("has_cyto", False))
        self.cyto_index = chan.get("cyto_index")
        self.use_cyto: bool = bool(seg.get("use_cyto", False)) and self.has_cyto
        if self.use_cyto and self.cyto_index is None:
            raise ValueError("channels.cyto_index must be set when use_cyto is true.")

        self.clip_percentile: float = float(seg.get("exclude_brightest_percentile", 99.9))
        self.flow_threshold: float = float(seg.get("flow_threshold", 0.4))
        self.cellprob_threshold: float = float(seg.get("cellprob_threshold", 0.0))
        self.min_mask_size: int = int(seg.get("min_mask_size", 15))
        self.batch_size: int = int(seg.get("batch_size", 8))
        # None → Cellpose-SAM (cpsam) default; else a path to a custom fine-tuned
        # model. Cellpose-SAM dropped the old built-in model-type names.
        self.pretrained_model = seg.get("pretrained_model")
        self._cfg_anisotropy = seg.get("anisotropy")  # None → from metadata

        self.adaptive_size_filter: bool = bool(seg.get("adaptive_size_filter", True))
        self.adaptive_size_k: float = float(seg.get("adaptive_size_k", 2.5))
        self.adaptive_dapi_filter: bool = bool(seg.get("adaptive_dapi_filter", False))
        self.adaptive_dapi_k: float = float(seg.get("adaptive_dapi_k", 2.5))

        vis = cfg.get("visualization", {})
        self.dpi: int = int(vis.get("dpi", 300))
        self.overlay_clip = tuple(vis.get("overlay_clip_percentile", (1, 99)))

        self.use_gpu: bool = _check_gpu()
        self._model = None

    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from cellpose import models
            logger.info("Segmentation: loading Cellpose model%s",
                        f" '{self.pretrained_model}'" if self.pretrained_model else " (Cellpose-SAM)")
            kwargs = {"gpu": self.use_gpu}
            if self.pretrained_model:
                kwargs["pretrained_model"] = self.pretrained_model
            self._model = models.CellposeModel(**kwargs)
        return self._model

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, path: Path, force: bool = False) -> None:
        from .image_loader import result_dir_for
        out_dir = result_dir_for(path)
        checkpoint = out_dir / CHECKPOINT
        if checkpoint.exists() and not force:
            logger.debug("Segmentation: %s already segmented.", path.name)
            return

        meta = load_image_metadata(out_dir)
        is_zstack = bool(meta.get("is_zstack", False))

        if self.mode == "3d" and not is_zstack:
            raise ValueError(
                f"{path.name}: 3-D segmentation requested but the image is a "
                f"MIP (no z-stack). Use segmentation.mode: 2d for this image."
            )

        dapi = self._load_channel(out_dir, self.dapi_index, is_zstack)
        cyto = None
        if self.use_cyto:
            cyto = self._load_channel(out_dir, int(self.cyto_index), is_zstack)

        logger.info(
            "Segmentation [%s]: %s%s",
            self.mode, path.name, " (DAPI+cyto)" if cyto is not None else "",
        )

        anisotropy = None
        if self.mode == "2d":
            masks, stats = self._run(dapi, cyto, do_3D=False)
            self._save_overlay_2d(dapi, masks, out_dir / "segmentation_overlay.png")
        else:
            anisotropy = self._get_anisotropy(meta)
            masks, stats = self._run(dapi, cyto, do_3D=True, anisotropy=anisotropy)
            self._save_overlay_3d(dapi, masks, out_dir)

        _save_size_filter_qc(stats, out_dir / "size_filter_qc.png", dpi=self.dpi)
        tifffile.imwrite(out_dir / "nuclear_masks.tif", masks.astype(np.uint32),
                         metadata={"axes": "ZYX" if self.mode == "3d" else "YX"})

        self._write_metadata(out_dir, stats, anisotropy)

        logger.info(
            "Segmentation [%s]: %s — %d nuclei (%d before filtering)",
            self.mode, path.name, stats["n_after"], stats["n_before"],
        )
        if self.use_gpu:
            import torch
            torch.cuda.empty_cache()
        checkpoint.touch()

    # ------------------------------------------------------------------
    # Cellpose execution
    # ------------------------------------------------------------------

    def _run(self, dapi, cyto, do_3D, anisotropy=None):
        """Clip the brightest pixels, run Cellpose, then filter masks."""
        dapi_c = self._clip(dapi)
        if cyto is not None:
            cyto_c = self._clip(cyto)
            img = np.stack([dapi_c, cyto_c], axis=-1)   # (...,2) channel-last
            channel_axis = -1
        else:
            img = dapi_c
            channel_axis = None

        eval_kwargs = dict(
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            do_3D=do_3D,
        )
        if channel_axis is not None:
            eval_kwargs["channel_axis"] = channel_axis
        if do_3D:
            eval_kwargs.update(z_axis=0, anisotropy=anisotropy, batch_size=self.batch_size)

        try:
            import torch
            ctx = getattr(torch, "inference_mode", torch.no_grad)
        except ImportError:
            import contextlib
            ctx = contextlib.nullcontext

        with ctx():
            masks, _, _ = self.model.eval(img, **eval_kwargs)

        from cellpose import utils as cp_utils
        masks = cp_utils.fill_holes_and_remove_small_masks(
            masks, min_size=self.min_mask_size
        )
        return self._filter_masks(masks, dapi_c)

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        """Exclude the brightest pixels by clipping at ``clip_percentile``."""
        arr = arr.astype(np.float32, copy=False)
        clip_val = np.percentile(arr, self.clip_percentile)
        return np.clip(arr, 0, clip_val)

    # ------------------------------------------------------------------
    # Adaptive mask filtering (shared by 2-D and 3-D; pixel/voxel agnostic)
    # ------------------------------------------------------------------

    def _filter_masks(self, masks: np.ndarray, dapi: np.ndarray):
        n_before = int(masks.max())
        if n_before == 0:
            return masks, _empty_stats(self.min_mask_size,
                                       self.adaptive_size_filter,
                                       self.adaptive_dapi_filter)

        counts = np.bincount(masks.ravel(), minlength=n_before + 1)
        areas = counts[1:n_before + 1].astype(np.float64)

        hard_min = float(self.min_mask_size)
        size_thr = hard_min
        remove_size = np.zeros(n_before, dtype=bool)
        if self.adaptive_size_filter and n_before >= _MIN_MASKS_FOR_ADAPTIVE:
            log_a = np.log(np.maximum(areas, 1.0))
            med = np.median(log_a)
            iqr_upper = max(float(np.percentile(log_a, 75)) - med, 1e-4)
            size_thr = max(hard_min, float(np.exp(med - self.adaptive_size_k * iqr_upper)))
            remove_size = areas < size_thr

        dapi_means = None
        dapi_thr = None
        remove_dapi = np.zeros(n_before, dtype=bool)
        if self.adaptive_dapi_filter and n_before >= _MIN_MASKS_FOR_ADAPTIVE:
            from scipy import ndimage as ndi
            labels = np.arange(1, n_before + 1, dtype=np.int32)
            dapi_means = np.asarray(
                ndi.mean(dapi.astype(np.float64), labels=masks, index=labels),
                dtype=np.float64,
            )
            log_d = np.log(np.maximum(dapi_means, 1e-6) + 1.0)
            med_d = np.median(log_d)
            iqr_upper_d = max(float(np.percentile(log_d, 75)) - med_d, 1e-4)
            dapi_thr = float(np.exp(med_d - self.adaptive_dapi_k * iqr_upper_d) - 1.0)
            remove_dapi = dapi_means < max(dapi_thr, 0.0)

        remove = remove_size | remove_dapi
        n_after = n_before - int(remove.sum())

        # Renumber surviving labels in a single lookup-table pass.
        keep = ~remove
        lut = np.zeros(n_before + 1, dtype=np.int32)
        lut[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
        masks_out = lut[masks]

        removed_by = np.full(n_before, "kept", dtype=object)
        removed_by[remove_size & ~remove_dapi] = "size"
        removed_by[~remove_size & remove_dapi] = "dapi"
        removed_by[remove_size & remove_dapi] = "both"

        if self.adaptive_size_filter or self.adaptive_dapi_filter:
            filter_type = "adaptive"
        else:
            filter_type = "hard_min"

        return masks_out, {
            "n_before": n_before,
            "n_after": n_after,
            "n_removed_size": int((remove_size & ~remove_dapi).sum()),
            "n_removed_dapi": int((~remove_size & remove_dapi).sum()),
            "n_removed_both": int((remove_size & remove_dapi).sum()),
            "hard_min": hard_min,
            "adaptive_size_threshold": size_thr,
            "dapi_threshold": dapi_thr,
            "filter_type": filter_type,
            "adaptive_size_filter": self.adaptive_size_filter,
            "adaptive_dapi_filter": self.adaptive_dapi_filter,
            "log2_areas": np.log2(np.maximum(areas, 1.0)),
            "dapi_means": dapi_means,
            "removed_by": removed_by,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_channel(self, out_dir: Path, ch_idx: int, is_zstack: bool) -> np.ndarray:
        ch_path = resolve_channel_path(out_dir, ch_idx, self.mode, is_zstack)
        if not ch_path.exists():
            raise FileNotFoundError(
                f"Channel file {ch_path.name} missing in {out_dir}. "
                f"Run the 'load' step first."
            )
        return tifffile.imread(ch_path).astype(np.float32, copy=False)

    def _get_anisotropy(self, meta: dict) -> float:
        if self._cfg_anisotropy is not None:
            return float(self._cfg_anisotropy)
        pz, px = meta.get("pixel_size_z_um"), meta.get("pixel_size_x_um")
        if pz and px and px > 0:
            aniso = float(pz) / float(px)
            logger.info("Segmentation [3D]: anisotropy from metadata = %.3f", aniso)
            return aniso
        logger.warning("Segmentation [3D]: anisotropy unknown — default %.1f",
                       _DEFAULT_ANISOTROPY)
        return _DEFAULT_ANISOTROPY

    def _write_metadata(self, out_dir: Path, stats: dict, anisotropy):
        meta = {
            "mode": self.mode,
            "model": self.pretrained_model or "cellpose-sam",
            "used_cyto_channel": self.use_cyto,
            "dapi_index": self.dapi_index,
            "cyto_index": int(self.cyto_index) if self.use_cyto else None,
            "anisotropy": anisotropy,
            "exclude_brightest_percentile": self.clip_percentile,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "min_mask_size": self.min_mask_size,
            "filter_type": stats["filter_type"],
            "adaptive_size_filter": stats["adaptive_size_filter"],
            "adaptive_size_threshold": stats["adaptive_size_threshold"],
            "adaptive_dapi_filter": stats["adaptive_dapi_filter"],
            "dapi_threshold": stats["dapi_threshold"],
            "n_nuclei_before_filtering": stats["n_before"],
            "n_nuclei_after_filtering": stats["n_after"],
            "n_removed_size": stats["n_removed_size"],
            "n_removed_dapi": stats["n_removed_dapi"],
            "n_removed_both": stats["n_removed_both"],
        }
        with open(out_dir / "segmentation_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _save_overlay_2d(self, dapi: np.ndarray, masks: np.ndarray, out_path: Path):
        rgb = _dapi_to_rgb(dapi, self.overlay_clip)
        rgb[_outlines_2d(masks)] = [1.0, 1.0, 0.0]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"Segmentation — {int(masks.max())} nuclei", fontsize=11)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _save_overlay_3d(self, dapi_stack: np.ndarray, masks_3d: np.ndarray, out_dir: Path):
        nz, ny, nx = dapi_stack.shape
        dapi_mip = dapi_stack.max(axis=0)
        n_cells = int(masks_3d.max())

        # (a) MIP overlay: each nucleus shown at its brightest Z plane.
        z_focus = dapi_stack.argmax(axis=0)
        mask_proj = masks_3d[z_focus, np.arange(ny)[:, None], np.arange(nx)]
        rgb = _dapi_to_rgb(dapi_mip, self.overlay_clip)
        rgb[_outlines_2d(mask_proj)] = [1.0, 1.0, 0.0]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"3-D segmentation MIP — {n_cells} nuclei", fontsize=11)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        fig.savefig(out_dir / "segmentation_overlay.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        # (b) Z-montage: up to 9 evenly spaced slices.
        n_panels = min(9, nz)
        z_idx = np.linspace(0, nz - 1, n_panels, dtype=int)
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5),
                                 gridspec_kw={"hspace": 0.05, "wspace": 0.05}, squeeze=False)
        for i, zi in enumerate(z_idx):
            ax = axes[i // ncols][i % ncols]
            rgb_sl = _dapi_to_rgb(dapi_stack[zi], self.overlay_clip)
            rgb_sl[_outlines_2d(masks_3d[zi])] = [1.0, 1.0, 0.0]
            ax.imshow(rgb_sl, interpolation="nearest")
            ax.set_title(f"z={zi}", fontsize=8, pad=2)
            ax.axis("off")
        for i in range(n_panels, nrows * ncols):
            axes[i // ncols][i % ncols].axis("off")
        fig.suptitle(f"3-D segmentation Z-montage — {n_cells} nuclei", fontsize=11)
        fig.savefig(out_dir / "segmentation_overlay_montage.png",
                    dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Module-level plotting / geometry helpers
# ---------------------------------------------------------------------------

def _dapi_to_rgb(dapi: np.ndarray, clip) -> np.ndarray:
    """Greyscale RGB image with extreme DAPI percentiles clipped."""
    vmin, vmax = np.percentile(dapi, clip)
    norm = np.clip((dapi.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return np.stack([norm] * 3, axis=-1)


def _outlines_2d(masks: np.ndarray) -> np.ndarray:
    """Vectorised outline detection for a 2-D label array."""
    out = np.zeros_like(masks, dtype=bool)
    out[:-1, :] |= masks[:-1, :] != masks[1:, :]
    out[1:, :] |= masks[:-1, :] != masks[1:, :]
    out[:, :-1] |= masks[:, :-1] != masks[:, 1:]
    out[:, 1:] |= masks[:, :-1] != masks[:, 1:]
    return out & (masks > 0)


_FILTER_COLORS = {"kept": "#2196F3", "size": "#F44336", "dapi": "#FF9800", "both": "#9C27B0"}
_FILTER_LABELS = {"kept": "Kept", "size": "Removed (size)",
                  "dapi": "Removed (DAPI)", "both": "Removed (both)"}


def _empty_stats(min_mask_size, adaptive_size, adaptive_dapi) -> dict:
    return {
        "n_before": 0, "n_after": 0,
        "n_removed_size": 0, "n_removed_dapi": 0, "n_removed_both": 0,
        "hard_min": float(min_mask_size),
        "adaptive_size_threshold": float(min_mask_size),
        "dapi_threshold": None,
        "filter_type": "adaptive" if (adaptive_size or adaptive_dapi) else "hard_min",
        "adaptive_size_filter": adaptive_size,
        "adaptive_dapi_filter": adaptive_dapi,
        "log2_areas": np.array([]),
        "dapi_means": None,
        "removed_by": np.array([], dtype=object),
    }


def _save_size_filter_qc(stats: dict, out_path: Path, dpi: int = 150):
    """Adaptive-filter diagnostic: log2(area) histogram (+ area-vs-DAPI scatter)."""
    n_before, n_after = stats["n_before"], stats["n_after"]
    n_removed = n_before - n_after

    has_dapi_panel = (
        stats["adaptive_dapi_filter"]
        and stats["dapi_means"] is not None
        and len(stats["dapi_means"]) > 0
    )
    ncols = 2 if has_dapi_panel else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5))
    axes = [axes] if ncols == 1 else list(axes)

    if n_before == 0:
        axes[0].text(0.5, 0.5, "No nuclei detected", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=11)
        axes[0].axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    log2_areas = stats["log2_areas"]
    removed_by = stats["removed_by"]

    ax1 = axes[0]
    n_bins = max(10, min(40, n_before // 5))
    edges = np.linspace(log2_areas.min() - 0.15, log2_areas.max() + 0.15, n_bins + 1)
    for cat in ("both", "size", "dapi", "kept"):
        m = removed_by == cat
        if m.any():
            ax1.hist(log2_areas[m], bins=edges, color=_FILTER_COLORS[cat], alpha=0.80,
                     label=f"{_FILTER_LABELS[cat]} (n={int(m.sum())})", edgecolor="none")

    log2_hard = np.log2(max(stats["hard_min"], 1.0))
    log2_adaptive = np.log2(max(stats["adaptive_size_threshold"], 1.0))
    ax1.axvline(log2_hard, ls="--", lw=1.2, color="#607D8B",
                label=f"hard floor ({int(stats['hard_min'])} px)")
    if stats["adaptive_size_filter"] and log2_adaptive > log2_hard + 0.02:
        ax1.axvline(log2_adaptive, ls="-", lw=1.8, color="#F44336",
                    label=f"adaptive ({stats['adaptive_size_threshold']:.0f} px)")
    pct = 100.0 * n_removed / max(n_before, 1)
    ax1.set_xlabel("log₂(mask area, px)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Mask size distribution\n{n_before} → {n_after} kept "
                  f"({n_removed} removed, {pct:.1f}%)", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.85)

    if has_dapi_panel:
        ax2 = axes[1]
        dapi_means = stats["dapi_means"]
        for cat in ("both", "size", "dapi", "kept"):
            m = removed_by == cat
            if m.any():
                ax2.scatter(log2_areas[m], dapi_means[m], c=_FILTER_COLORS[cat],
                            s=8, alpha=0.55, linewidths=0,
                            label=f"{_FILTER_LABELS[cat]} (n={int(m.sum())})")
        if stats["adaptive_size_filter"] and log2_adaptive > log2_hard + 0.02:
            ax2.axvline(log2_adaptive, ls="-", lw=1.4, color="#F44336", alpha=0.75)
        ax2.axvline(log2_hard, ls="--", lw=1.0, color="#607D8B", alpha=0.7)
        if stats["dapi_threshold"] is not None:
            ax2.axhline(stats["dapi_threshold"], ls="-", lw=1.4, color="#FF9800",
                        alpha=0.85, label=f"DAPI floor ({stats['dapi_threshold']:.1f})")
        ax2.set_xlabel("log₂(mask area, px)")
        ax2.set_ylabel("Mean DAPI intensity")
        ax2.set_title("Area vs. mean DAPI (filter outcome)", fontsize=10)
        ax2.legend(fontsize=8, framealpha=0.85)

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
