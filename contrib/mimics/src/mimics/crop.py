"""Crop butterfly wings out of a Cambridge segfolder dataset.

Reads an ImgSegFolder dataset and writes a new ImgSegFolder dataset with the
same structure (images/{split}/, annotations/{split}/, labels.csv) where each
image is cropped to the bounding box of the non-background mask and the
remaining background is filled with a uniform color. Masks are cropped with
the same bounding box so they stay pixel-aligned with the images, which keeps
patch-label generation in the shard pipeline correct.
"""

import concurrent.futures
import dataclasses
import pathlib
import shutil
import typing as tp

import beartype
import numpy as np
from PIL import Image

from saev.data.datasets import ImgSegFolder

DEFAULT_SRC_DPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/datasets/cambridge-segfolder-v1.6"
)
DEFAULT_OUT_DPATH = pathlib.Path(
    "/fs/ess/PAS2136/jbeattie/datasets/cambridge-segfolder-cropped"
)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Crop butterfly wings out of a Cambridge segfolder dataset."""

    src_dpath: pathlib.Path = DEFAULT_SRC_DPATH
    """Source segfolder root."""
    out_dpath: pathlib.Path = DEFAULT_OUT_DPATH
    """Output segfolder root."""
    split: tp.Literal["training", "validation"] = "training"
    """Which split to crop (training or validation)."""
    bg_label: int = 0
    """Mask value treated as background."""
    bg_fill: tuple[int, int, int] = (0, 0, 0)
    """RGB color to fill background pixels with."""
    pad: int = 0
    """Padding in pixels to add around the foreground bounding box."""
    n_threads: int = 16
    """Number of concurrent worker threads."""
    labels_csv: str = "labels.csv"
    """Labels CSV filename, copied verbatim from the source."""


@beartype.beartype
def _foreground_bbox(mask: np.ndarray, *, bg_label: int) -> tuple[int, int, int, int]:
    """Bounding box of non-background pixels as (left, top, right, bottom).

    The right and bottom edges are exclusive. Raises on an all-background mask.
    """
    fg_rows, fg_cols = np.where(mask != bg_label)
    msg = f"Mask has no foreground pixels (all background label {bg_label})."
    assert fg_rows.size > 0, msg
    return (
        int(fg_cols.min()),
        int(fg_rows.min()),
        int(fg_cols.max()) + 1,
        int(fg_rows.max()) + 1,
    )


@beartype.beartype
def _padded_bbox(
    bbox: tuple[int, int, int, int], *, pad: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """Expand a bbox by pad, clamping to the image bounds."""
    left, top, right, bottom = bbox
    return (
        max(left - pad, 0),
        max(top - pad, 0),
        min(right + pad, w),
        min(bottom + pad, h),
    )


@beartype.beartype
def _fill_background(
    img: Image.Image, mask: Image.Image, *, fill: tuple[int, int, int]
) -> Image.Image:
    """Set pixels where the mask is background to a uniform fill color."""
    mask_nd = np.asarray(mask, dtype=np.uint8)
    img_nd = np.array(img.convert("RGB"), dtype=np.uint8)
    img_nd[mask_nd == 0] = np.asarray(fill, dtype=np.uint8)
    return Image.fromarray(img_nd)


@beartype.beartype
def _crop_one(
    cfg: Config, *, img_fpath: pathlib.Path, seg_fpath: pathlib.Path
) -> tuple[int, int]:
    """Crop one image and its mask. Returns (n_written, n_skipped)."""
    stem = img_fpath.stem
    out_img_fpath = cfg.out_dpath / "images" / cfg.split / f"{stem}.jpg"
    out_seg_fpath = cfg.out_dpath / "annotations" / cfg.split / f"{stem}.png"

    if out_img_fpath.exists() and out_seg_fpath.exists():
        return 0, 1

    img = Image.open(img_fpath).convert("RGB")
    seg = Image.open(seg_fpath).convert("L")
    seg_nd = np.asarray(seg)

    bbox = _foreground_bbox(seg_nd, bg_label=cfg.bg_label)
    left, top, right, bottom = _padded_bbox(
        bbox, pad=cfg.pad, w=img.width, h=img.height
    )

    img = img.crop((left, top, right, bottom))
    seg = seg.crop((left, top, right, bottom))
    img = _fill_background(img, seg, fill=cfg.bg_fill)

    img.save(out_img_fpath)
    seg.save(out_seg_fpath)
    return 1, 0


@beartype.beartype
def worker_fn(cfg: Config) -> None:
    """Crop all images in the source segfolder into the output segfolder."""
    src_img_dpath = cfg.src_dpath / "images" / cfg.split
    src_seg_dpath = cfg.src_dpath / "annotations" / cfg.split
    assert src_img_dpath.is_dir(), f"Source images dir missing: '{src_img_dpath}'."
    assert src_seg_dpath.is_dir(), f"Source annotations dir missing: '{src_seg_dpath}'."

    out_img_dpath = cfg.out_dpath / "images" / cfg.split
    out_seg_dpath = cfg.out_dpath / "annotations" / cfg.split
    out_img_dpath.mkdir(parents=True, exist_ok=True)
    out_seg_dpath.mkdir(parents=True, exist_ok=True)

    src_labels_fpath = cfg.src_dpath / cfg.labels_csv
    assert src_labels_fpath.is_file(), (
        f"Source labels CSV missing: '{src_labels_fpath}'."
    )
    shutil.copyfile(src_labels_fpath, cfg.out_dpath / cfg.labels_csv)

    seg_by_stem = {p.stem: p for p in src_seg_dpath.iterdir()}
    img_fpaths = sorted(src_img_dpath.iterdir())

    n_written = 0
    n_skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futures = [
            pool.submit(
                _crop_one,
                cfg,
                img_fpath=img_fpath,
                seg_fpath=seg_by_stem[img_fpath.stem],
            )
            for img_fpath in img_fpaths
            if img_fpath.stem in seg_by_stem
        ]
        for future in concurrent.futures.as_completed(futures):
            written, skipped = future.result()
            n_written += written
            n_skipped += skipped

    assert n_written + n_skipped == len(img_fpaths), (
        f"Cropped {n_written + n_skipped} of {len(img_fpaths)} images."
    )

    out_cfg = ImgSegFolder(
        root=cfg.out_dpath, split=cfg.split, labels_csv=cfg.labels_csv
    )
    n_out = out_cfg.n_examples
    assert n_out == len(img_fpaths), (
        f"Output has {n_out} images, expected {len(img_fpaths)}."
    )

    print(
        f"Cropped {n_written} images (skipped {n_skipped} already-written) "
        f"from '{cfg.src_dpath}' to '{cfg.out_dpath}'."
    )
