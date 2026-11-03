"""Run original-source JBU vs equation (4)-(7) PCJBF on official scenes."""
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / '.vendor'))

import cv2
import numpy as np
import numba
from experiment.dataset import load_scene, make_lr
from experiment.algorithms import source_jbu, pcjbf
from experiment.metrics import evaluate
from experiment.visualize import make_scene_figures
from experiment.paths import RESULTS_ROOT

NAMES = ['Baby', 'Books', 'Bowling', 'Cloth', 'Dolls', 'Laundry', 'Moebius', 'Reindeer', 'Wood']
FIELDS = ['scene', 'factor', 'jbu_bpr', 'pcjbf_bpr', 'jbu_rmse', 'pcjbf_rmse',
          'jbu_seconds', 'pcjbf_seconds', 'valid_pixels', 'p1_pct', 'p2_pct', 'p3_pct', 'p4_pct', 'roi']


def run_one(name, factor, output):
    rgb, gt, mask, _ = load_scene(name)
    lr = make_lr(gt, factor)
    t0 = time.perf_counter()
    baseline = source_jbu(rgb, lr, factor)
    jbu_time = time.perf_counter() - t0
    base_dir = output / 'numeric' / f'{name}_x{factor}'
    base_dir.mkdir(parents=True, exist_ok=True)
    np.save(base_dir / 'jbu.npy', baseline)  # saved before PCJBF begins
    jbu_bpr, jbu_rmse, n = evaluate(baseline, gt, mask)
    print(f'{name} x{factor}: baseline saved, BPR {jbu_bpr:.3f}%, RMSE {jbu_rmse:.3f}', flush=True)

    t0 = time.perf_counter()
    ours, depth_edge, color_edge, classes = pcjbf(rgb, lr, baseline, factor)
    ours_time = time.perf_counter() - t0
    ours_bpr, ours_rmse, _ = evaluate(ours, gt, mask)
    roi = make_scene_figures(output, name, factor, rgb, gt, lr, baseline, ours,
                             mask, depth_edge, color_edge, classes)
    proportions = [float(np.mean(classes == i) * 100) for i in range(1, 5)]
    return dict(zip(FIELDS, [name, factor, jbu_bpr, ours_bpr, jbu_rmse, ours_rmse,
                             jbu_time, ours_time, n, *proportions, json.dumps(roi)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=NAMES)
    parser.add_argument('--factors', nargs='+', type=int, default=[2, 4])
    args = parser.parse_args()
    out = RESULTS_ROOT
    out.mkdir(exist_ok=True)
    upstream_commit = 'e4b0b207c3da0b8772b94c89b1138c3fa956159d'
    provenance = {
        'upstream_commit': upstream_commit,
        'python': platform.python_version(), 'opencv_python': cv2.__version__,
        'numpy': np.__version__, 'numba': numba.__version__,
        'dataset_manifest': json.loads((out / 'dataset/middlebury/manifest.json').read_text(encoding='utf-8')),
        'parameters': {'source_jbu_sigma_s': 1, 'source_jbu_sigma_r': 10,
                       'pcjbf_sigma_s': .2, 'pcjbf_sigma_r': .1,
                       'depth_variance_threshold': .005, 'color_edge_threshold': .10,
                       'bpr_bad_if_abs_error_gt': 1.0,
                       'lr_interpolation': 'nearest', 'gt_mask': 'finite and >0'},
    }
    (out / 'run_manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_file = out / 'raw_metrics.csv'
    existing = []
    if csv_file.exists():
        with csv_file.open(newline='', encoding='utf-8') as f:
            existing = list(csv.DictReader(f))
    scenes = [args.scene] if args.scene else NAMES
    for factor in args.factors:
        if factor not in (2, 4, 8):
            raise ValueError('factor must be 2, 4, or 8')
        for name in scenes:
            if any(r['scene'] == name and int(r['factor']) == factor for r in existing):
                print('Already done:', name, factor, flush=True)
                continue
            row = run_one(name, factor, out)
            existing.append(row)
            with csv_file.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, FIELDS)
                writer.writeheader()
                writer.writerows(existing)
            print(f"{name} x{factor}: PCJBF BPR {row['pcjbf_bpr']:.3f}%, RMSE {row['pcjbf_rmse']:.3f}", flush=True)
    print('Finished:', len(existing), 'scene-factor pairs')


if __name__ == '__main__':
    main()
