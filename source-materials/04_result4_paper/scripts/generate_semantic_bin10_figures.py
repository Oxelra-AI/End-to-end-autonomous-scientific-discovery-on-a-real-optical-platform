#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the original semantic benchmark figures at bin=10.
This directly answers the operator request: rerun the previously used
bin=25 paper/report figures at speckle-matched bin=10 for comparison.
"""

import os
import json
import numpy as np
from analyze_complex_B_semantic import (
    load_single_token_data, load_blank_control_data, load_pair_interaction_data,
    compute_interaction_observables, compute_per_repeat_B, build_feature_families,
    evaluate_all, evaluate_cross_split, analyze_complex_B_stability, make_figures
)
import analyze_complex_B_semantic as mod


def main():
    section_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_dir = os.path.dirname(section_dir)
    data_dir = os.path.join(package_dir, 'shared_raw_data', 'result4_frame_data',
                            'exp_complex_B_semantic')
    out_dir = os.path.join(section_dir, 'source_data', 'processed',
                           'complex_B_semantic_results_bin10')
    fig_dir = os.path.join(section_dir, 'figures', 'semantic_bin10')
    os.makedirs(fig_dir, exist_ok=True)

    # Override global BIN_SIZE in imported module
    mod.BIN_SIZE = 10
    mod.N_CHANNELS = (mod.ROI_H // mod.BIN_SIZE) * (mod.ROI_W // mod.BIN_SIZE)

    print(f'Running semantic benchmark figure regeneration at BIN_SIZE={mod.BIN_SIZE}')
    print(f'N_CHANNELS={mod.N_CHANNELS}')

    single_data = load_single_token_data(data_dir)
    blank_data = load_blank_control_data(data_dir)
    pair_data = load_pair_interaction_data(data_dir)

    Q_pair, Q_blank, B_pair, z_single = compute_interaction_observables(single_data, blank_data, pair_data)
    B_per_rep = compute_per_repeat_B(blank_data, pair_data)
    stability_info = analyze_complex_B_stability(B_per_rep)

    families, pair_labels, cat_binary_labels, cat_pair_labels = build_feature_families(z_single, B_per_rep)
    results = evaluate_all(families, pair_labels, cat_binary_labels, cat_pair_labels)
    cross_results, held_out_pairs = evaluate_cross_split(families, cat_binary_labels, pair_labels)

    make_figures(results, cross_results, B_pair, z_single, B_per_rep,
                 pair_data, stability_info, fig_dir,
                 cmos1_dir=os.path.join(data_dir, 'cmos1_documentation'))

    payload = {
        'bin_size': 10,
        'n_channels': mod.N_CHANNELS,
        'results': results,
        'cross_results': cross_results,
        'stability_mean': stability_info[1],
        'stability_min': stability_info[2],
        'held_out_pairs': held_out_pairs,
    }
    with open(os.path.join(out_dir, 'complex_B_semantic_results_bin10.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    print('Saved results and figures to', out_dir)


if __name__ == '__main__':
    main()
