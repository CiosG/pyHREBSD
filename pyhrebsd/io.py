"""Image loading shared by the command-line and configured analysis runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_pattern(path: str | Path) -> np.ndarray:
    """Read grayscale/RGB image and crop a rectangular detector to a square."""
    path = Path(path)
    with Image.open(path) as image:
        data = np.asarray(image, dtype=np.float64)
    if data.ndim == 3:
        data = data[..., :3].mean(axis=2)
    if data.ndim != 2:
        raise ValueError(f"{path} is not a grayscale or RGB image")
    height, width = data.shape
    if width > height:
        start = (width - height) // 2
        data = data[:, start : start + height]
    elif height > width:
        start = (height - width) // 2
        data = data[start : start + width, :]
    return data
