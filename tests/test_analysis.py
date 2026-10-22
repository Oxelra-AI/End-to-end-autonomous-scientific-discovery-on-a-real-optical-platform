"""Scientific invariants of the maintained evaluation and readout code."""

import unittest

import numpy as np

from analysis.budgets import nested_random_budget
from analysis.phase_readout import closure, demodulate
from analysis.result4 import category_splits, evaluate, grouped_splits, scale_pair


class AnalysisTests(unittest.TestCase):
    def test_scaling_uses_training_statistics(self):
        train = np.array([[1., 7.], [3., 7.], [5., 7.]])
        test = np.array([[1000., -20.]])
        a, b = scale_pair(train, test)
        np.testing.assert_allclose(a[:, 0].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(a[:, 0].std(), 1)
        np.testing.assert_array_equal(a[:, 1], 0)
        np.testing.assert_allclose(b, [[997 / np.std([1., 3., 5.]), -27.]])
        np.testing.assert_array_equal(train, [[1., 7.], [3., 7.], [5., 7.]])

    def test_category_holdout_excludes_tokens_and_self_pairs(self):
        info = np.array([(x, y) for x in range(8) for y in range(8) for _ in range(5)])
        for held, (tr, te) in enumerate(category_splits(info, True, True)):
            self.assertEqual(len(tr), 150)
            self.assertEqual(len(te), 130)
            self.assertFalse(np.any(info[tr] // 2 == held))
            self.assertTrue(np.all(np.any(info[te] // 2 == held, axis=1)))
            self.assertTrue(np.all(info[tr, 0] != info[tr, 1]))
            self.assertTrue(np.all(info[te, 0] != info[te, 1]))

    def test_unordered_pair_groups_hold_both_orders_together(self):
        pairs = np.array([(x, y) for x in range(4) for y in range(4) for _ in range(5)])
        groups = pairs.min(axis=1) * 4 + pairs.max(axis=1)
        for tr, te in grouped_splits(groups):
            train_pairs = {tuple(sorted(p)) for p in pairs[tr]}
            test_pairs = {tuple(sorted(p)) for p in pairs[te]}
            self.assertFalse(train_pairs & test_pairs)

    def test_unseen_identity_labels_are_not_scored_as_closed_set(self):
        with self.assertRaisesRegex(ValueError, 'Closed-set'):
            evaluate(np.arange(8).reshape(4, 2), np.array([0, 0, 1, 1]),
                     [(np.array([0, 1]), np.array([2, 3]))])

    def test_nested_selection_does_not_use_outer_test_label(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(26, 12))
        y = np.arange(26) % 2
        outer = [(np.arange(24), np.array([24, 25]))]
        first = nested_random_budget(X, y, 4, outer_splits=outer)
        changed = y.copy()
        changed[-2:] = 1 - changed[-2:]
        second = nested_random_budget(X, changed, 4, outer_splits=outer)
        self.assertEqual(first['selected_candidate_by_fold'], second['selected_candidate_by_fold'])
        self.assertEqual(first['predictions'], second['predictions'])

    def test_phase_closure_and_demodulation_for_fixed_fields(self):
        rng = np.random.default_rng(7)
        a, b, reference = rng.normal(size=(3, 30)) + 1j * rng.normal(size=(3, 30))
        phases = np.arange(4) * np.pi / 2
        pair = np.array([abs(reference + a + np.exp(1j * phi) * b)**2 for phi in phases])
        blank = np.array([abs(reference + np.exp(1j * phi) * b)**2 for phi in phases])
        np.testing.assert_allclose(closure(pair), 0, atol=2e-14)
        np.testing.assert_allclose(demodulate(pair) - demodulate(blank), np.conj(a) * b, atol=2e-14)

    def test_phase_dependent_gain_is_visible_to_closure(self):
        fields = np.array([1., 1., 0., 1.])
        self.assertEqual(closure(fields), -1.)


if __name__ == '__main__':
    unittest.main()
