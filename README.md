# Confocal Image-Processing Pipeline

A modular, format-agnostic pipeline for processing **multichannel confocal
microscopy** images. It discovers image files anywhere under a directory and,
for each one, converts the raw acquisition into analysis-ready arrays, segments
nuclei with **Cellpose-SAM**, and extracts per-nucleus intensities leaving
all downstream / experiment-specific analysis to the individual user.

The pipeline is designed for shared lab use: point it at a folder, run it, and
every image gets its own self-contained results directory that you can consume
in your own analyses.

---

## What it does

For every input image, three independently runnable steps execute in order:

| Step | Command | Produces (per image) |
| --- | --- | --- |
| **load** | `--steps load` | `image_metadata.json`, `multichannel.tif`, `ch{n}.tif` (+ `*_MIP.tif` for z-stacks) |
| **segment** | `--steps segment` | `nuclear_masks.tif`, `segmentation_overlay.png`, `size_filter_qc.png`, `segmentation_metadata.json` |
| **extract** | `--steps extract` | `per_nuclei_intensities.joblib` |

Each step writes a hidden checkpoint (`.loaded`, `.segmented`, `.extracted`) so
re-running the pipeline only redoes what is missing.

### Key features

- **Any file type** — `.nd2`, `.tif`/`.tiff`, `.ome.tif`, `.oir`, `.czi`,
  `.lif`, and anything else `bioio` can read. Backends are auto-selected per
  file; axis order is normalized automatically.
- **MIP vs z-stack auto-detection** — the loader inspects each image and writes
  the appropriate outputs. Z-stacks additionally get per-channel maximum
  intensity projections.
- **2-D or 3-D segmentation** — 2-D runs on the DAPI MIP (works for any input);
  3-D runs on the DAPI z-stack with anisotropy taken from the metadata.
- **Optional cytoplasm channel** — DAPI + cytoplasm can be passed together to
  Cellpose-SAM for joint segmentation.
- **Adaptive nucleus filtering** — removes debris/oversplit masks using a
  robust size (and optional DAPI) distribution estimator, with a QC figure.
- **Per-nucleus quantification** — raw and DAPI-normalized intensities plus
  nucleus **centroids** (2-D or 3-D) for locating saturated/outlier signal.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**GPU (recommended for segmentation):** install a CUDA build of PyTorch that
matches your driver *before* the rest, e.g.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Extra file formats:** the universal `bioio` backend needs a plugin per
format — e.g. `pip install bioio-czi` for `.czi`, `pip install bioio-bioformats`
(plus a Java runtime) for `.oir`. `.nd2` and TIFF variants work out of the box.

---

## Configuration

All behaviour is driven by [`config.yaml`](config.yaml). The essentials:

```yaml
input:
  directory: ./data            # folder to search (recursively)
  file_types: [.nd2, .tif, .oir, ...]

channels:
  dapi_index: 0                # DAPI channel index
  has_cyto: false              # is there a cytoplasm channel?
  cyto_index: 1                # its index
  marker_channels: [2, 3]      # channels to quantify

segmentation:
  mode: 2d                     # 2d | 3d | null  (null → 2-D on DAPI MIP)
  use_cyto: false              # include cyto channel in segmentation
  min_mask_size: 15
  adaptive_size_filter: true
  anisotropy: null             # 3-D: null → from metadata, fallback 2.0

intensity:
  normalize_by_dapi: true
```

Channel indices are **0-based** and refer to the channel axis of your image.

---

## Usage

```bash
# Everything: load → segment → extract, for every image under input.directory
python main.py

# Only convert/load images (e.g. to inspect channels first)
python main.py --steps load

# Segmentation + extraction (assumes load already ran)
python main.py --steps segment extract

# 3-D segmentation on z-stacks, overriding the config
python main.py --seg-mode 3d

# Recompute a single step for all images
python main.py --force-step segment

# Recompute everything from scratch
python main.py --force
```

Useful flags: `--config <path>`, `--log-level DEBUG`. A full run log is also
written to `pipeline.log`. Failures on one image are logged with a full
traceback and do not stop the remaining images.

---

## Outputs

Each image `path/to/img.nd2` produces a sibling directory `path/to/img/`:

```
img/
├── image_metadata.json          # pixel sizes, channels, MIP-vs-stack, axes, ...
├── multichannel.tif             # CYX (MIP) or CZYX (z-stack)
├── ch0.tif, ch1.tif, ...        # per-channel images
├── ch0_MIP.tif, ...             # per-channel MIPs (z-stacks only)
├── multichannel_MIP.tif         # (z-stacks only)
├── nuclear_masks.tif            # label image (2-D, or z-stack for 3-D mode)
├── segmentation_overlay.png     # nuclei outlined on DAPI (+ _montage for 3-D)
├── size_filter_qc.png           # adaptive filter diagnostic
├── segmentation_metadata.json   # parameters + nucleus counts before/after
└── per_nuclei_intensities.joblib
```

### Reading the intensity results

```python
import joblib
r = joblib.load("path/to/img/per_nuclei_intensities.joblib")

r["n_cells"]                       # number of nuclei
r["labels"]                        # mask label id per nucleus
r["centroids"]                     # (n, 2) [y,x] or (n, 3) [z,y,x]
r["centroid_axes"]                 # "yx" or "zyx"
r["mask_sizes"]                    # pixels (2-D) / voxels (3-D) per nucleus
r["dapi_per_cell"]                 # per-nucleus DAPI level
r["raw_intensities"][2]            # per-nucleus median of channel 2
r["dapi_normalized_intensities"][2]  # channel 2 / DAPI per nucleus
r["channel_names"]                 # {index: name}
r["global_stats"][2]               # mean/median/percentiles for channel 2
```

Arrays are aligned row-for-row by `labels`, so you can build a per-nucleus table
(intensities + centroids + sizes) directly for your downstream analysis.

---

## Project structure

```
.
├── main.py                       # CLI: discovery + per-image step orchestration
├── config.yaml                   # all pipeline parameters
├── requirements.txt
├── run_3d_segmentation.lsf       # Example script for running 3D (or 2D) segmentation/pipeline on the CCHMC Cluster
└── src/image_processing/
    ├── io.py                     # versatile multi-format reader → canonical CZYX
    ├── directory_walker.py       # recursive, output-aware image discovery
    ├── image_loader.py           # MIP/stack detection, channel TIFFs, metadata
    ├── segmentation.py           # Cellpose-SAM 2-D/3-D + adaptive filter + QC
    ├── intensity_extractor.py    # per-nucleus intensities + centroids
    └── utils.py                  # shared metadata / channel-path helpers
```

### Extending the pipeline

- **New file format** — add a `_read_<fmt>` backend in `io.py` returning
  `(data, axes, meta)` and register its extension; everything downstream is
  format-agnostic.
- **New per-image step** — add a class with a `process(path, force=False)`
  method and a checkpoint file, then register it in `STEP_ORDER` /
  `build_steps()` in `main.py`.

The core helpers (`io.read_image`, vectorized label statistics, adaptive
filtering, overlays) are written to be reusable across any image-analysis task.
