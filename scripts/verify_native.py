"""Cross-check native C++ output against the batch experiment pixel by pixel."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / '.vendor'))
from experiment.paths import RESULTS_ROOT
import cv2
import numpy as np

SCENES = [('Baby', 'Baby1', 2), ('Wood', 'Wood1', 4), ('Books', 'Books', 8)]


def main():
    candidates = [ROOT / 'build_native/pcjbf_experiment.exe',
                  ROOT / 'build/pcjbf_experiment.exe',
                  ROOT / 'build/Release/pcjbf_experiment.exe']
    exe = next((path for path in candidates if path.exists()), None)
    if exe is None:
        raise FileNotFoundError('Build pcjbf_experiment before native verification')
    records = []
    for label, scene, factor in SCENES:
        stem = f'{label}_x{factor}'
        output = RESULTS_ROOT / 'validation' / f'native_{label.lower()}_x{factor}'
        output.mkdir(parents=True, exist_ok=True)
        process = subprocess.run([str(exe), str(RESULTS_ROOT / f'dataset/middlebury/{scene}/view1.png'),
                                  str(RESULTS_ROOT / f'dataset/middlebury/{scene}/disp1.png'),
                                  str(factor), str(output)], cwd=ROOT, capture_output=True,
                                 text=True, check=True)
        counts = {}
        for method in ('jbu', 'pcjbf'):
            expected = np.load(RESULTS_ROOT / 'numeric' / stem / f'{method}.npy')
            actual = cv2.imread(str(output / f'{method}.tiff'), cv2.IMREAD_UNCHANGED)
            if actual is None or actual.shape != expected.shape:
                raise AssertionError(f'{stem} {method} unreadable or wrong shape')
            counts[method] = int(np.count_nonzero(expected != actual))
        expected_class = np.load(RESULTS_ROOT / 'numeric' / stem / 'pixel_classes.npy')
        actual_class = cv2.imread(str(output / 'classes.png'), cv2.IMREAD_UNCHANGED)
        counts['classes'] = int(np.count_nonzero(expected_class != actual_class))
        records.append({'scene': label, 'factor': factor, 'different_pixels': counts,
                        'native_stdout': process.stdout.strip()})
        if any(counts.values()):
            raise AssertionError(f'Native/Python mismatch: {stem} {counts}')
    destination = RESULTS_ROOT / 'validation/native_validation.json'
    destination.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Native/Python exact match for {len(records)} cases. Saved {destination}')


if __name__ == '__main__':
    main()
