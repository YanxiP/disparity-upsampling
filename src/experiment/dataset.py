"""Middlebury full-size aligned view1/disp1 reader; preserves numeric disparity."""
from pathlib import Path
import json
import cv2
import numpy as np

from .paths import RESULTS_ROOT

DATA_ROOT = RESULTS_ROOT / 'dataset/middlebury'


def read_disparity(path: Path) -> np.ndarray:
    """Read PNG as stored or PFM float pixels, retaining invalid values.

    PFM scanlines are bottom-up and the scale sign encodes endianness.
    """
    path = Path(path)
    if path.suffix.lower() != '.pfm':
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None or array.ndim != 2:
            raise ValueError(f'Cannot read single-channel disparity: {path}')
        return array.astype(np.float32)
    with path.open('rb') as f:
        header = f.readline().strip()
        if header != b'Pf':
            raise ValueError(f'Expected grayscale PFM (Pf): {path}')
        dimensions = f.readline().strip()
        while dimensions.startswith(b'#'):
            dimensions = f.readline().strip()
        width, height = map(int, dimensions.split())
        scale = float(f.readline().strip())
        dtype = '<f4' if scale < 0 else '>f4'
        pixels = np.fromfile(f, dtype=dtype, count=width*height)
    if pixels.size != width*height:
        raise ValueError(f'Truncated PFM: {path}')
    return np.flipud(pixels.reshape(height, width)).astype(np.float32) * abs(scale)


def load_scene(label: str, multiple: int = 8):
    manifest = json.loads((DATA_ROOT / 'manifest.json').read_text(encoding='utf-8'))
    item = manifest[label]
    root = DATA_ROOT
    rgb = cv2.imread(str(root / item['rgb']), cv2.IMREAD_COLOR)
    disparity = read_disparity(root / item['disparity'])
    if rgb is None:
        raise ValueError(f'Cannot read aligned scene {label}')
    if rgb.shape[:2] != disparity.shape:
        raise ValueError(f'Unaligned RGB/disparity in {label}: {rgb.shape} vs {disparity.shape}')
    h, w = disparity.shape
    h -= h % multiple
    w -= w % multiple
    rgb = rgb[:h, :w].copy()
    gt = disparity[:h, :w].copy()
    valid = np.isfinite(gt) & (gt > 0)
    return rgb, gt, valid, item


def make_lr(gt: np.ndarray, factor: int):
    h, w = gt.shape
    assert h % factor == 0 and w % factor == 0
    # Exactly the nearest-neighbor downsampling stated in the PCJBF experiment.
    return cv2.resize(gt, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)
