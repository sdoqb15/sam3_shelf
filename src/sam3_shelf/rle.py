"""Lossless C-order binary RLE; this is not COCO's compressed/column-major RLE."""

import numpy as np


def encode_rle(mask):
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    if not flat.size:
        raise ValueError("Cannot encode an empty image")
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, boundaries, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return counts


def decode_rle(counts, height, width):
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or (values < 0).any() or int(values.sum()) != height * width:
        raise ValueError("RLE counts must be nonnegative and sum to height * width")
    return np.repeat(np.arange(len(values)) % 2, values).astype(bool).reshape(height, width)
