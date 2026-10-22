"""Four-phase readout consistency in the released camera-array units.

No per-phase gains are fitted. The closure condition assumes fixed fields and
reference, accurate phase steps, and a common linear intensity calibration.
"""

import argparse
from pathlib import Path

import numpy as np

from .result4 import dump, load_frames


def demodulate(intensities):
    """Phases are ordered as 0, pi/2, pi, 3pi/2, matching the released code."""
    i0, i1, i2, i3 = np.asarray(intensities)
    return (i0 - i2 + 1j * (i3 - i1)) / 4


def closure(intensities):
    i0, i1, i2, i3 = np.asarray(intensities)
    return i0 + i2 - i1 - i3


def describe(pair, blank):
    mean = pair.mean(axis=1)
    residual = closure(mean)
    total = mean.sum(axis=0)
    norm = float(np.linalg.norm(total))
    total_mean = float(total.mean())
    return {
        'phase_roi_means': mean.mean(axis=1).tolist(),
        'phase_roi_repeat_std': pair.mean(axis=2).std(axis=1, ddof=1).tolist(),
        'closure_roi_by_repeat': closure(pair).mean(axis=1).tolist(),
        'closure_roi_over_sum_roi': float(residual.mean() / total_mean) if total_mean else None,
        'closure_l2_over_sum_l2': float(np.linalg.norm(residual) / norm) if norm else None,
        'blank_closure_roi_mean': float(closure(blank).mean()),
        'blank_subtracted_closure_roi_mean': float(closure(pair - blank).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deposit', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'source-materials')
    parser.add_argument('--output', type=Path, default=Path('outputs/phase_readout'))
    args = parser.parse_args()
    dep, out = args.deposit.resolve(), args.output.resolve()
    if out == dep or dep in out.parents:
        parser.error('Choose an output directory outside the archived source materials.')
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'frame_cache'
    cache.mkdir(exist_ok=True)
    results = {
        'units': 'released calibrated array units; no additional gain correction',
        'condition': 'I0 + Ipi = Ipi2 + I3pi2 for fixed fields, reference and common intensity calibration',
        'interpretation': 'A closure residual checks this necessary model condition; it is not a relative error estimate for B.',
        'experiments': {},
    }
    for experiment, nt in [('exp_complex_B_semantic', 8), ('exp_bilinear_001', 4)]:
        _, blank, pair = load_frames(dep, experiment, 25, nt, cache)
        rows = []
        for x in range(nt):
            for y in range(nt):
                p = np.array([pair[x, y, phase] for phase in range(4)])
                b = np.array([blank[y, phase] for phase in range(4)])
                rows.append({'x': x, 'y': y, **describe(p, b)})
        results['experiments'][experiment] = rows
        print(experiment, 'pairs:', len(rows), flush=True)
    dump(out / 'phase_readout.json', results)


if __name__ == '__main__':
    main()
