"""Feature-budget comparisons with training-fold preprocessing and selection."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from threadpoolctl import threadpool_limits

from .result4 import dump, evaluate, scale_pair


def mean_random_budget(X, y, budget, n_random=10, seed=42):
    """Average prespecified random subsets; do not select by test performance."""
    if budget < 1 or n_random < 1:
        raise ValueError('budget and n_random must be positive')
    rng = np.random.default_rng(seed)
    subsets = ([np.arange(X.shape[1])] if budget >= X.shape[1] else
               [rng.choice(X.shape[1], budget, replace=False) for _ in range(n_random)])
    splits = list(LeaveOneOut().split(X))
    scores = [evaluate(X[:, cols], y, splits, 'train')['balanced_accuracy']
              for cols in subsets]
    return {'mean_balanced_accuracy': float(np.mean(scores)),
            'subset_scores': scores, 'subset_indices': [s.tolist() for s in subsets]}


def nested_random_budget(X, y, budget, n_random=5, seed=42, outer_splits=None):
    """Select columns using inner CV; use each outer test fold only once.

    The default outer unit is one measurement. This estimates recognition of
    measured input combinations, as specified in EVALUATION.md.
    """
    if budget < 1 or n_random < 1:
        raise ValueError('budget and n_random must be positive')
    rng = np.random.default_rng(seed)
    candidates = ([np.arange(X.shape[1])] if budget >= X.shape[1] else
                  [rng.choice(X.shape[1], budget, replace=False) for _ in range(n_random)])
    if outer_splits is None:
        outer_splits = LeaveOneOut().split(X)
    truth, predictions, test_indices, selection = [], [], [], []
    for tr, te in outer_splits:
        tr, te = np.asarray(tr), np.asarray(te)
        if not len(tr) or not len(te) or np.intersect1d(tr, te).size:
            raise ValueError('Training and test indices must be nonempty and disjoint')
        if len(candidates) == 1:
            chosen = 0
        else:
            _, counts = np.unique(y[tr], return_counts=True)
            n_inner = min(3, int(counts.min()))
            if n_inner < 2:
                raise ValueError('Inner CV needs at least two training examples per class')
            inner = list(StratifiedKFold(n_splits=n_inner, shuffle=True,
                                        random_state=seed).split(X[tr], y[tr]))
            scores = []
            for cols in candidates:
                fold_scores = []
                for a, b in inner:
                    xa, xb = scale_pair(X[tr[a]][:, cols], X[tr[b]][:, cols])
                    model = RidgeClassifier(alpha=1.0).fit(xa, y[tr[a]])
                    fold_scores.append(balanced_accuracy_score(y[tr[b]], model.predict(xb)))
                scores.append(float(np.mean(fold_scores)))
            chosen = int(np.argmax(scores))
        cols = candidates[chosen]
        xa, xb = scale_pair(X[tr][:, cols], X[te][:, cols])
        pred = RidgeClassifier(alpha=1.0).fit(xa, y[tr]).predict(xb)
        truth.extend(y[te].tolist())
        predictions.extend(pred.tolist())
        test_indices.extend(te.tolist())
        selection.append(chosen)
    return {'balanced_accuracy': float(balanced_accuracy_score(truth, predictions)),
            'truth': truth, 'predictions': predictions, 'test_indices': test_indices,
            'selected_candidate_by_fold': selection,
            'candidate_indices': [c.tolist() for c in candidates]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('outputs/result4'))
    parser.add_argument('--output', type=Path, default=Path('outputs/budgets'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    threadpool_limits(limits=1)
    semantic = np.load(args.features / 'semantic_bin25_features.npz', allow_pickle=False)
    result = {}
    for family in ['zx_zy_concat', 'digital_bilinear_ew', 'complex_B']:
        result[family] = {}
        for task in 'ABC':
            result[family][task] = mean_random_budget(
                semantic[family], semantic[f'label_{task}'], 50)
            print('semantic', family, task,
                  result[family][task]['mean_balanced_accuracy'], flush=True)
    dump(args.output / 'semantic_k50.json', result)
    xor = np.load(args.features / 'xor_bin25_features.npz', allow_pickle=False)
    result = {}
    for family in ['zx_zy_concat', 'complex_B']:
        result[family] = {}
        for task in 'AB':
            result[family][task] = nested_random_budget(xor[family], xor[f'label_{task}'], 400)
            print('xor', family, task, result[family][task]['balanced_accuracy'], flush=True)
    dump(args.output / 'xor_nested_k400.json', result)


if __name__ == '__main__':
    main()
