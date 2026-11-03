"""Reusable figures with one display scale per scene and shared error limits."""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _limits(gt, valid):
    vals = gt[valid]
    return float(vals.min()), float(vals.max())


def _depth(ax, array, title, lo, hi):
    ax.imshow(np.ma.masked_where(array <= 0, array), cmap='viridis', vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def roi_from_gt(gt, valid, size=170):
    """Pick strongest GT boundary, independent of either algorithm's output."""
    good = gt.copy()
    good[~valid] = 0
    gx = cv2.Sobel(good, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(good, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge[~valid] = 0
    edge = cv2.boxFilter(edge, -1, (size, size), normalize=False)
    margin = size // 2
    edge[:margin] = 0
    edge[-margin:] = 0
    edge[:, :margin] = 0
    edge[:, -margin:] = 0
    cy, cx = np.unravel_index(edge.argmax(), edge.shape)
    return int(cy-margin), int(cy+margin), int(cx-margin), int(cx+margin)


def make_scene_figures(root, name, factor, rgb, gt, lr, jbu, ours,
                       valid, depth_edge, color_edge, classes):
    root = Path(root) / 'visualizations'
    stem = f'{name}_x{factor}'
    lo, hi = _limits(gt, valid)
    rgb_show = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    lr_show = cv2.resize(lr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)
    axes[0].imshow(rgb_show); axes[0].set_title('HR RGB'); axes[0].axis('off')
    for ax, ar, title in zip(axes[1:], (gt, lr_show, jbu, ours),
                             ('Ground truth', f'LR input x{factor}', 'Original JBU', 'PCJBF')):
        _depth(ax, ar, title, lo, hi)
    _save(fig, root / 'full_comparison' / f'{stem}_comparison.png')

    y0, y1, x0, x1 = roi_from_gt(gt, valid)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), constrained_layout=True)
    axes[0].imshow(rgb_show[y0:y1, x0:x1], interpolation='nearest')
    axes[0].set_title('HR RGB')
    for ax, ar, title in zip(axes[1:], (gt, jbu, ours), ('Ground truth', 'Original JBU', 'PCJBF')):
        _depth(ax, ar[y0:y1, x0:x1], title, lo, hi)
    for ax in axes: ax.axis('off')
    fig.suptitle(f'{stem} | same GT-edge ROI: x={x0}:{x1}, y={y0}:{y1}', fontsize=11)
    _save(fig, root / 'edge_zoom' / f'{stem}_edge_zoom.png')

    err_a = np.abs(jbu - gt)
    err_b = np.abs(ours - gt)
    err_a[~valid] = np.nan
    err_b[~valid] = np.nan
    error_arrays = root.parent / 'numeric' / 'errors'
    error_arrays.mkdir(parents=True, exist_ok=True)
    np.save(error_arrays / f'{stem}_JBU_error.npy', err_a)
    np.save(error_arrays / f'{stem}_PCJBF_error.npy', err_b)
    upper = max(float(np.nanpercentile(err_a, 99)), float(np.nanpercentile(err_b, 99)), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, arr, title in zip(axes, (err_a, err_b), ('JBU absolute error', 'PCJBF absolute error')):
        im = ax.imshow(arr, cmap='inferno', vmin=0, vmax=upper)
        ax.set_title(title); ax.axis('off')
    fig.colorbar(im, ax=axes, label=f'Absolute disparity error (clipped at shared P99={upper:.1f} px)', shrink=.65)
    _save(fig, root / 'error_maps' / f'{stem}_error_comparison.png')

    cmap = ListedColormap(['#e53e3e', '#3182ce', '#ed8936', '#718096'])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    axes[0].imshow(rgb_show); axes[0].set_title('HR RGB')
    axes[1].imshow(depth_edge, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Depth edge')
    axes[2].imshow(color_edge, cmap='gray', vmin=0, vmax=1); axes[2].set_title('Color edge')
    axes[3].imshow(classes, cmap=cmap, vmin=.5, vmax=4.5)
    axes[3].set_title('P1 red | P2 blue | P3 orange | P4 gray')
    for ax in axes: ax.axis('off')
    _save(fig, root / 'classification_maps' / f'{stem}_pixel_classes.png')

    numeric = Path(root).parent / 'numeric' / stem
    numeric.mkdir(parents=True, exist_ok=True)
    np.save(numeric / 'ground_truth.npy', gt)
    np.save(numeric / 'lr_input.npy', lr)
    np.save(numeric / 'jbu.npy', jbu)
    np.save(numeric / 'pcjbf.npy', ours)
    np.save(numeric / 'valid_mask.npy', valid)
    np.save(numeric / 'pixel_classes.npy', classes)
    return (x0, y0, x1, y1)
