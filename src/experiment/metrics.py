import numpy as np


def evaluate(prediction, ground_truth, mask, threshold=1.0):
    """BPR percent for absolute disparity error > 1 pixel; RMSE in pixels.

    The PCJBF paper does not state its numeric BPR threshold in the PDF; this
    reproducible convention is therefore an explicit assumption.
    """
    good = mask & np.isfinite(prediction) & np.isfinite(ground_truth)
    diff = np.abs(prediction[good].astype(np.float64) - ground_truth[good].astype(np.float64))
    if not diff.size:
        raise ValueError('No valid ground truth pixels')
    return float(100 * np.mean(diff > threshold)), float(np.sqrt(np.mean(diff ** 2))), int(diff.size)
