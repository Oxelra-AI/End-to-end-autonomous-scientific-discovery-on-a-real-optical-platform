"""Collect compact tables from completed offline runs without copying camera data."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from .result4 import dump


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def score_record(record):
    keys = ('balanced_accuracy', 'mean_fold_balanced_accuracy', 'folds')
    return {key: record[key] for key in keys if key in record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('outputs'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    runs, out = args.runs, args.output
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for experiment in ('semantic', 'xor'):
        main_results = read(runs / 'result4' / f'{experiment}_main.json')
        splits = read(runs / 'result4' / f'{experiment}_split_comparisons.json')
        summary[experiment] = {'measurement_loo': {}, 'additional_splits': {}}
        for family, tasks in main_results.items():
            summary[experiment]['measurement_loo'][family] = {
                task: {'archived': value['published'], 'regenerated': value['reproduced'],
                       'training_fold_normalization': value['corrected']['balanced_accuracy']}
                for task, value in tasks.items()
            }
        for family, tasks in splits.items():
            compact = {}
            for task, protocols in tasks.items():
                if task == 'D_fixed16':
                    compact[task] = {'held_pairs': protocols['held_pairs'],
                                     'archived': protocols['published'],
                                     'corrected': score_record(protocols['corrected'])}
                else:
                    compact[task] = {protocol: score_record(record)
                                     for protocol, record in protocols.items()}
            summary[experiment]['additional_splits'][family] = compact
    dump(out / 'main_summary.json', summary)
    raw = read(runs / 'refined' / 'raw_baselines.json')
    dump(out / 'raw_input_baselines.json', {
        family: {'archived': values['published'], 'regenerated': values['reproduced'],
                 'D_no_self': score_record(values['D_no_self']),
                 'D_nonself_test_only': score_record(values['D_nonself_test_only'])}
        for family, values in raw.items()
    })
    semantic = read(runs / 'budgets' / 'semantic_k50.json')
    xor = read(runs / 'budgets' / 'xor_nested_k400.json')
    dump(out / 'feature_budgets.json', {
        'semantic_k50_mean10': {family: {task: record['mean_balanced_accuracy']
                                        for task, record in tasks.items()}
                                for family, tasks in semantic.items()},
        'xor_k400_nested5': {family: {task: record['balanced_accuracy']
                                      for task, record in tasks.items()}
                              for family, tasks in xor.items()},
    })
    for source, target in [
        ('result4/environment.json', 'main_environment.json'),
        ('refined/environment.json', 'refined_environment.json'),
        ('result4/main_comparison.csv', 'main_comparison.csv'),
        ('refined/sparse_summary.csv', 'sparse_summary.csv'),
        ('phase_readout/phase_readout.json', 'phase_readout.json'),
    ]:
        shutil.copyfile(runs / source, out / target)
    sparse = read(runs / 'refined' / 'sparse_comparisons.json')
    grid_checks = []
    for record in sparse:
        for task in 'ABCD':
            if record['published'][task] != record['reproduced'][task]:
                grid_checks.append([record['bin'], record['n_out'], record['family'], task])
    with (out / 'main_comparison.csv').open(newline='') as stream:
        comparisons = list(csv.DictReader(stream))
    unchanged = sum(abs(float(r['reproduced']) - float(r['corrected'])) < 1e-12
                    for r in comparisons)
    dump(out / 'reproduction_summary.json', {
        'main_scores': len(comparisons),
        'main_scores_unchanged_by_training_fold_normalization': unchanged,
        'maximum_archived_main_difference': max(
            abs(float(r['published']) - float(r['reproduced'])) for r in comparisons),
        'sparse_scores': 4 * len(sparse),
        'sparse_mismatches_at_archived_precision': grid_checks,
        'interpretation': 'Reproduction compares the released scores at their stored precision; added holdouts answer different prediction questions.',
    })
    print('Wrote compact results to', out)


if __name__ == '__main__':
    main()
