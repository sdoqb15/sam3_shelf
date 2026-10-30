"""Native-size overlays for human review, independent of training targets."""

import numpy as np
from PIL import Image

COLORS = (
    (242, 60, 60),
    (60, 113, 242),
    (166, 242, 60),
    (242, 60, 219),
    (60, 242, 211),
    (242, 180, 60),
)


def overlay(image, masks):
    rgb = np.asarray(image.convert("RGB")).copy()
    for index, mask in enumerate(masks):
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.asarray(COLORS[index % len(COLORS)])
    return Image.fromarray(rgb)
