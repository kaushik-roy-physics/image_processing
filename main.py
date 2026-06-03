#!/usr/bin/env python3
"""
Confocal image-processing pipeline — lab-wide, format-agnostic.

Discovers multichannel microscope images under a directory and, for each file,
runs three independently checkpointed steps:

  load     →  read image, detect MIP vs z-stack, write channel TIFFs + metadata
  segment  →  Cellpose-SAM nuclear segmentation (2-D or 3-D) + QC overlays
  extract  →  per-nucleus intensities (raw + DAPI-normalised) + centroids

Each step writes its outputs into a per-image result directory and drops a
checkpoint file so re-runs skip completed work. Use --force / --force-step to
recompute.

Usage examples
--------------
  python main.py                                  # all steps, all images
  python main.py --steps load                     # just load/convert images
  python main.py --steps segment extract          # segmentation + extraction
  python main.py --seg-mode 3d                     # override config seg mode
  python main.py --force                           # recompute everything
  python main.py --force-step segment              # recompute one step
  python main.py --config my_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ---------------------------------------------------------------------------
# Config / logging
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str, log_file: str = "pipeline.log") -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    return logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

# Ordered: each step depends on outputs of the ones before it.
STEP_ORDER = ["load", "segment", "extract"]


def build_steps(cfg: dict):
    from image_processing import ImageLoader, IntensityExtractor, Segmentation
    return {
        "load": ImageLoader(cfg),
        "segment": Segmentation(cfg),
        "extract": IntensityExtractor(cfg),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confocal image-processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    p.add_argument("--steps", nargs="+", choices=STEP_ORDER, default=STEP_ORDER,
                   help="Steps to run, in order (default: all)")
    p.add_argument("--seg-mode", choices=["2d", "3d"], default=None, dest="seg_mode",
                   help="Override segmentation.mode from the config")
    p.add_argument("--force", action="store_true",
                   help="Recompute all selected steps, ignoring checkpoints")
    p.add_argument("--force-step", nargs="+", choices=STEP_ORDER, default=[],
                   dest="force_step", help="Recompute specific step(s) only")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log = setup_logging(args.log_level)

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return 1

    cfg = load_config(config_path)
    cfg["_root"] = str(config_path.parent.resolve())
    if args.seg_mode is not None:
        cfg.setdefault("segmentation", {})["mode"] = args.seg_mode

    steps_to_run = [s for s in STEP_ORDER if s in set(args.steps)]
    force_steps = set(args.force_step)

    from image_processing import DirectoryWalker
    try:
        walker = DirectoryWalker(cfg)
        images = walker.find_images()
    except Exception:
        log.exception("Failed to discover input images")
        return 1

    if not images:
        log.error("No images to process. Exiting.")
        return 1

    steps = build_steps(cfg)
    seg_mode = (cfg.get("segmentation", {}).get("mode") or "2d").lower()
    log.info("=== Pipeline start — %d image(s), steps=%s, seg_mode=%s ===",
             len(images), steps_to_run, seg_mode)

    n_ok, n_failed = 0, 0
    for i, path in enumerate(images, 1):
        rel = _safe_relpath(path, cfg["_root"])
        log.info("[%d/%d] %s", i, len(images), rel)
        try:
            for step in steps_to_run:
                force = args.force or (step in force_steps)
                steps[step].process(path, force=force)
            n_ok += 1
        except Exception:
            # Log full traceback but continue with remaining images.
            log.exception("Failed processing %s", rel)
            n_failed += 1

    log.info("=== Pipeline finished — %d ok, %d failed ===", n_ok, n_failed)
    return 1 if n_failed else 0


def _safe_relpath(path: Path, root: str) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
