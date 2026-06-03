"""
Versatile multi-format microscopy image reader.

A single entry point, :func:`read_image`, loads any supported microscope file
(``.nd2``, ``.tif/.tiff``, ``.ome.tif/.ome.tiff``, ``.oir``, ``.czi``, ``.lif``,
…) and returns a canonical ``(C, Z, Y, X)`` (or ``(C, Y, X)`` for single-plane
images) NumPy array together with a format-agnostic metadata dictionary.

Backend dispatch
----------------
* ``.nd2``                          → ``nd2``       (fast, rich metadata)
* ``.tif / .tiff / .ome.tif(f)``    → ``tifffile``  (axes from series metadata)
* everything else (``.oir`` …)      → ``bioio``     (universal fallback)

Adding a new native backend only requires writing a ``_read_<fmt>`` function
that returns ``(data, axes, meta)`` and registering its extension below.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Extensions handled by a dedicated, dependency-light backend. Everything else
# is routed to the universal bioio backend (which may pull in Bio-Formats).
_TIFF_EXTS = {".tif", ".tiff", ".ome.tif", ".ome.tiff"}
_NATIVE_EXTS = {".nd2"} | _TIFF_EXTS

# Default set advertised to users / config. bioio extends this transparently.
SUPPORTED_EXTENSIONS = sorted(_NATIVE_EXTS | {".oir", ".czi", ".lif"})

# Axis letters we understand. Anything else (P positions, etc.) is reduced to
# its first index so we always converge on a (C, [Z], Y, X) array.
_KNOWN_AXES = set("TCZYXS")


def get_extension(path: Path) -> str:
    """Return a lower-case extension, treating ``.ome.tif(f)`` as one unit."""
    name = path.name.lower()
    if name.endswith(".ome.tiff"):
        return ".ome.tiff"
    if name.endswith(".ome.tif"):
        return ".ome.tif"
    return path.suffix.lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_image(path: Path) -> tuple[np.ndarray, dict]:
    """
    Load ``path`` and return ``(data, meta)``.

    ``data`` is ``(C, Z, Y, X)`` when the image is a z-stack, otherwise
    ``(C, Y, X)``. ``meta`` always contains the keys ``is_zstack``,
    ``n_channels``, ``n_z``, ``axes_canonical``, ``file_format`` and
    ``source_file`` plus any pixel-size / channel-name information the backend
    could recover.
    """
    path = Path(path)
    ext = get_extension(path)

    if ext == ".nd2":
        data, axes, meta = _read_nd2(path)
    elif ext in _TIFF_EXTS:
        data, axes, meta = _read_tiff(path)
    else:
        data, axes, meta = _read_bioio(path)

    data, has_z = _to_canonical(data, axes)

    n_z = int(data.shape[1]) if has_z else 1
    meta.setdefault("file_format", ext)
    meta["source_file"] = path.name
    meta["n_channels"] = int(data.shape[0])
    meta["n_z"] = n_z
    meta["is_zstack"] = bool(has_z and n_z > 1)
    meta["axes_canonical"] = "CZYX" if meta["is_zstack"] else "CYX"

    # A degenerate z-axis (n_z == 1) is squeezed away so downstream code only
    # ever sees true stacks as 4-D.
    if has_z and n_z == 1:
        data = data[:, 0]

    return data, meta


# ---------------------------------------------------------------------------
# Canonicalisation: arbitrary labelled array → (C, [Z], Y, X)
# ---------------------------------------------------------------------------

def _to_canonical(data: np.ndarray, axes: str) -> tuple[np.ndarray, bool]:
    """
    Reorder a labelled array to ``(C, Z, Y, X)`` (or ``(C, Y, X)``).

    Returns ``(array, has_z)``. Unknown / singleton acquisition axes (time,
    stage position, …) are collapsed to their first index.
    """
    axes = axes.upper()
    if len(axes) != data.ndim:
        # Backend mislabelled the array — fall back to a shape heuristic.
        return _heuristic_canonical(data)

    # 1. Drop time and any unknown leading axes (take first index).
    for ax in list(axes):
        if ax in ("T",) or ax not in _KNOWN_AXES:
            idx = axes.index(ax)
            data = np.take(data, 0, axis=idx)
            axes = axes[:idx] + axes[idx + 1:]

    # 2. RGB "samples" axis (S) becomes the channel axis if no explicit C.
    if "S" in axes:
        if "C" in axes:
            si = axes.index("S")
            data = np.take(data, 0, axis=si)
            axes = axes.replace("S", "")
        else:
            axes = axes.replace("S", "C")

    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"Image is missing spatial axes (axes={axes!r}).")

    # 3. Move present axes into canonical C, Z, Y, X order.
    order = [a for a in "CZYX" if a in axes]
    data = np.transpose(data, [axes.index(a) for a in order])
    axes = "".join(order)

    has_z = "Z" in axes
    if "C" not in axes:
        data = data[np.newaxis, ...]   # promote to single channel

    return np.ascontiguousarray(data), has_z


def _heuristic_canonical(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Last-resort axis inference from shape alone (unlabelled arrays)."""
    data = np.squeeze(data)
    if data.ndim == 2:                       # (Y, X)
        return data[np.newaxis], False
    if data.ndim == 3:
        # Small leading dim → channels (MIP); otherwise a single-channel stack.
        if data.shape[0] <= 6:
            return data, False               # (C, Y, X)
        return data[np.newaxis], True        # (1, Z, Y, X)
    if data.ndim == 4:
        # Assume (Z, C, Y, X) if the second dim is the smaller of the two.
        z, c = data.shape[0], data.shape[1]
        if c <= z:
            data = np.transpose(data, (1, 0, 2, 3))   # → (C, Z, Y, X)
        return data, True
    raise ValueError(f"Cannot interpret array with shape {data.shape}.")


# ---------------------------------------------------------------------------
# Backend: nd2
# ---------------------------------------------------------------------------

def _read_nd2(path: Path) -> tuple[np.ndarray, str, dict]:
    import nd2

    with nd2.ND2File(path) as f:
        data = f.asarray()
        axes = "".join(f.sizes.keys())   # asarray follows f.sizes order
        meta: dict = {"file_format": ".nd2"}
        try:
            a = f.attributes
            meta.update(width_px=a.widthPx, height_px=a.heightPx,
                        n_channels=a.channelCount)
        except Exception:
            pass
        try:
            v = f.voxel_size()
            meta.update(pixel_size_x_um=float(v.x),
                        pixel_size_y_um=float(v.y),
                        pixel_size_z_um=float(v.z))
        except Exception:
            pass
        try:
            chans = f.metadata.channels
            if chans:
                meta["channel_names"] = [str(c.channel.name) for c in chans]
        except Exception:
            pass
    return data, axes, meta


# ---------------------------------------------------------------------------
# Backend: tifffile (plain TIFF + OME-TIFF + ImageJ)
# ---------------------------------------------------------------------------

def _read_tiff(path: Path) -> tuple[np.ndarray, str, dict]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes                      # e.g. 'ZCYX', 'CYX', 'QYX'
        meta: dict = {"file_format": get_extension(path)}

        # 'Q'/'I' are tifffile's "unknown"/"sequence" labels. Resolve to a
        # channel axis when small, else a z-axis, so canonicalisation succeeds.
        axes = _resolve_unknown_tiff_axes(axes, data.shape)

        pixel = _tiff_pixel_sizes(tif)
        meta.update(pixel)
        names = _tiff_channel_names(tif)
        if names:
            meta["channel_names"] = names
        if data.ndim >= 2:
            meta["height_px"], meta["width_px"] = int(data.shape[-2]), int(data.shape[-1])
    return data, axes, meta


def _resolve_unknown_tiff_axes(axes: str, shape: tuple) -> str:
    out = []
    for ax, size in zip(axes, shape):
        if ax in _KNOWN_AXES:
            out.append(ax)
        elif ax in ("Q", "I"):
            # Heuristic: a small unknown axis is channels, a large one is Z.
            out.append("C" if (size <= 6 and "C" not in out) else "Z")
        else:
            out.append("C" if size <= 6 else "Z")
    return "".join(out)


def _tiff_pixel_sizes(tif) -> dict:
    out: dict = {}
    # ImageJ stores z-spacing; XY come from resolution tags.
    ij = getattr(tif, "imagej_metadata", None) or {}
    if "spacing" in ij:
        out["pixel_size_z_um"] = float(ij["spacing"])
    try:
        page = tif.pages[0]
        tags = page.tags
        for axis, tag in (("x", "XResolution"), ("y", "YResolution")):
            if tag in tags:
                val = tags[tag].value
                res = val[0] / val[1] if isinstance(val, tuple) else float(val)
                if res:
                    out[f"pixel_size_{axis}_um"] = 1.0 / res
    except Exception:
        pass
    # OME-TIFF physical sizes (authoritative when present).
    ome = getattr(tif, "ome_metadata", None)
    if ome:
        import re
        for axis in ("X", "Y", "Z"):
            m = re.search(rf'PhysicalSize{axis}="([0-9.eE+-]+)"', ome)
            if m:
                out[f"pixel_size_{axis.lower()}_um"] = float(m.group(1))
    return out


def _tiff_channel_names(tif) -> list[str] | None:
    ome = getattr(tif, "ome_metadata", None)
    if ome:
        import re
        names = re.findall(r'<Channel[^>]*Name="([^"]+)"', ome)
        if names:
            return names
    return None


# ---------------------------------------------------------------------------
# Backend: bioio (universal fallback — .oir, .czi, .lif, …)
# ---------------------------------------------------------------------------

def _read_bioio(path: Path) -> tuple[np.ndarray, str, dict]:
    try:
        from bioio import BioImage
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"Reading '{get_extension(path)}' files requires the 'bioio' "
            "backend and a matching plugin (e.g. 'bioio-bioformats' for .oir, "
            "'bioio-czi' for .czi, 'bioio-lif' for .lif). "
            "Install with: pip install bioio bioio-bioformats"
        ) from exc

    img = BioImage(path)
    data = np.asarray(img.data)     # bioio is always 5-D TCZYX
    meta: dict = {"file_format": get_extension(path)}
    try:
        ps = img.physical_pixel_sizes
        if ps.X:
            meta["pixel_size_x_um"] = float(ps.X)
        if ps.Y:
            meta["pixel_size_y_um"] = float(ps.Y)
        if ps.Z:
            meta["pixel_size_z_um"] = float(ps.Z)
    except Exception:
        pass
    try:
        if img.channel_names:
            meta["channel_names"] = [str(c) for c in img.channel_names]
    except Exception:
        pass
    return data, "TCZYX", meta
