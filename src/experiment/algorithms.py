"""Faithful source-logic JBU baseline and explicit equation (4)-(7) PCJBF integration."""
from __future__ import annotations
import cv2
import numpy as np
from numba import njit


@njit(cache=True)
def _source_joint_filter(guide, disp, sigma_s=1.0, sigma_r=10.0):
    """Original main.cpp joint_bilateral_filter loop, fixed 5x5 for sigma_s=1.

    Original code uses uchar guide/disp, Gaussian spatial/range weights, rounds
    to uchar. This implementation preserves those operations and edge handling.
    """
    h, w = disp.shape
    out = np.empty((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            numerator = 0.0
            denominator = 0.0
            center = float(guide[y, x])
            for dy in range(-2, 3):
                yy = y + dy
                if yy < 0 or yy >= h:
                    continue
                for dx in range(-2, 3):
                    xx = x + dx
                    if xx < 0 or xx >= w:
                        continue
                    d = center - float(guide[yy, xx])
                    weight = np.exp(-(dx*dx + dy*dy) / (2*sigma_s*sigma_s) - d*d / (2*sigma_r*sigma_r))
                    numerator += weight * float(disp[yy, xx])
                    denominator += weight
            out[y, x] = np.uint8(np.floor(numerator / denominator + 0.5))
    return out


def source_jbu(rgb, lr, factor: int):
    """Reproduce upstream upsample(1,10,disp,gray): resize then joint filter.

    The upstream executable cannot run headlessly and its OpenCV C++ development
    dependency is absent here. Its function body is retained in src/main.cpp.
    """
    guide = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    current = np.clip(np.floor(lr + .5), 0, 255).astype(np.uint8)
    stages = int(np.floor(np.log2(factor)))
    for _ in range(1, stages - 1):
        current = cv2.resize(current, (current.shape[1] * 2, current.shape[0] * 2))
        intermediate = cv2.resize(guide, (current.shape[1], current.shape[0]))
        current = _source_joint_filter(intermediate, current)
    current = cv2.resize(current, (guide.shape[1], guide.shape[0]))
    return _source_joint_filter(guide, current).astype(np.float32)


def edge_maps(rgb, lr, factor: int, depth_threshold=.005, color_threshold=.10):
    """Eq. (4),(5): LR normalized local variance, HR local gray max-min.

    The paper omits threshold values and normalized-variance denominator. We use
    5x5 valid-only variance divided by the squared valid disparity range.
    """
    valid = (lr > 0).astype(np.float32)
    n = cv2.boxFilter(valid, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REFLECT)
    summ = cv2.boxFilter(lr * valid, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(lr * lr * valid, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(sq / np.maximum(n, 1) - (summ / np.maximum(n, 1)) ** 2, 0)
    vals = lr[valid > 0]
    denom = max(float(vals.max() - vals.min()) ** 2, 1.0)
    depth_lr = (variance / denom >= depth_threshold) & (n >= 2)
    depth_hr = np.repeat(np.repeat(depth_lr, factor, 0), factor, 1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    element = np.ones((5, 5), np.uint8)
    color_hr = (cv2.dilate(gray, element).astype(np.float32) -
                cv2.erode(gray, element).astype(np.float32)) / 255 >= color_threshold
    return depth_hr, color_hr


def classify(depth_edge, color_edge):
    """Eq. (6): values 1..4 correspond to P1..P4."""
    out = np.full(depth_edge.shape, 4, np.uint8)
    out[~depth_edge & color_edge] = 3
    out[depth_edge & ~color_edge] = 2
    out[depth_edge & color_edge] = 1
    return out


@njit(cache=True)
def _refine(guide, lr, initial, classes, factor, sigma_s=.2, sigma_r=.1):
    h, w = initial.shape
    lh, lw = lr.shape
    out = initial.copy()
    for y in range(h):
        ly = y // factor
        for x in range(w):
            lx = x // factor
            cls = classes[y, x]
            best_score = -1.0e30 if cls == 1 else 1.0e30
            best_depth = initial[y, x]
            # Eq. (7): P2 uses four adjacent LR samples (Omega); P1/P3/P4
            # use the 5x5 LR neighborhood N(p_L).
            ymin = max(0, ly - (0 if cls == 2 else 2))
            ymax = min(lh, ly + (2 if cls == 2 else 3))
            xmin = max(0, lx - (0 if cls == 2 else 2))
            xmax = min(lw, lx + (2 if cls == 2 else 3))
            for qy in range(ymin, ymax):
                for qx in range(xmin, xmax):
                    candidate = lr[qy, qx]
                    if candidate <= 0 or not np.isfinite(candidate):
                        continue
                    if cls == 1:
                        # Eq. (7), case I. Project q to HR sample center.
                        hy = min(h - 1, qy * factor + factor // 2)
                        hx = min(w - 1, qx * factor + factor // 2)
                        distance = np.sqrt((ly - qy)**2 + (lx - qx)**2) / 2.0
                        color_delta = abs(float(guide[y, x]) - float(guide[hy, hx])) / 255.0
                        score = np.exp(-distance / (2 * sigma_s) - color_delta / (2 * sigma_r)) * candidate
                        if score > best_score:
                            best_score = score
                            best_depth = candidate
                    else:
                        # Eq. (7), cases II and III: argmin |D_L(q)-D_JBU(p)|.
                        score = abs(candidate - initial[y, x])
                        if score < best_score:
                            best_score = score
                            best_depth = candidate
            out[y, x] = best_depth
    return out


def pcjbf(rgb, lr, initial, factor: int):
    depth_edge, color_edge = edge_maps(rgb, lr, factor)
    classes = classify(depth_edge, color_edge)
    guide = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    refined = _refine(guide, lr, initial, classes, factor)
    return refined, depth_edge, color_edge, classes
