"""
DirectoryWalker: recursively discover microscope image files under a root.

The walker is format-agnostic — it simply collects every file whose extension
is in the configured ``file_types`` set. Crucially, it never re-ingests its own
outputs: each per-image result directory created by :class:`ImageLoader`
contains a hidden marker file (:data:`OUTPUT_MARKER`), and any directory holding
that marker is pruned from the traversal. This keeps the discovery step
idempotent even though the channel ``.tif`` outputs share an extension with
potential ``.tif`` inputs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .io import SUPPORTED_EXTENSIONS, get_extension

logger = logging.getLogger(__name__)

# Dropped into every per-image output directory so we can skip it on re-walks.
OUTPUT_MARKER = ".imgproc_output"


class DirectoryWalker:
    def __init__(self, cfg: dict):
        ipt = cfg.get("input", {})
        root = ipt.get("directory")
        if not root:
            raise ValueError("config: 'input.directory' must be set.")
        self.root = Path(root).expanduser().resolve()
        self.recursive: bool = bool(ipt.get("recursive", True))
        exts = ipt.get("file_types") or SUPPORTED_EXTENSIONS
        self.extensions = {e.lower() if e.startswith(".") else f".{e.lower()}"
                           for e in exts}

    def find_images(self) -> list[Path]:
        """Return a sorted list of input image paths, excluding pipeline outputs."""
        if not self.root.exists():
            raise FileNotFoundError(f"Input directory not found: {self.root}")

        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune output directories (and, if non-recursive, all sub-dirs).
            if not self.recursive and Path(dirpath) != self.root:
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if not (Path(dirpath) / d / OUTPUT_MARKER).exists()
            ]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if get_extension(fpath) in self.extensions:
                    found.append(fpath)

        found.sort()
        logger.info(
            "DirectoryWalker: %d image file(s) under %s (types: %s)",
            len(found), self.root, ", ".join(sorted(self.extensions)),
        )
        if not found:
            logger.warning(
                "DirectoryWalker: no matching images found — check "
                "'input.directory' and 'input.file_types' in the config."
            )
        return found
