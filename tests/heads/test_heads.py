"""Frozen likelihood calibration and safe persistence across swappable heads."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prediction_heads.registry import create_head, load_head, manifest


class PredictionHeadTests(unittest.TestCase):
    def test_empirical_tail_ties_extremes_and_strict_cutoff(self):
        head = create_head().fit([3., 1., 2., 2.])
        scored = head.score([-100., 1., 1.5, 2., 3., 100.])
        np.testing.assert_allclose(scored['tail_probability'], [.2, .4, .4, .8, 1., 1.])
        np.testing.assert_allclose(scored['score'], -np.log2(scored['tail_probability']))
        # Equality to the cutoff is allowed, including a tied score group.
        self.assertFalse((scored['score'] > head.get_threshold(.2)).any())
        np.testing.assert_array_equal(scored['score'] > head.get_threshold(.4),
                                      [True, False, False, False, False, False])
        tied = create_head().fit([0., 0., 0.]).score([0.])
        self.assertEqual(tied['tail_probability'][0], 1.)

    def test_reference_is_frozen_and_no_query_changes_it(self):
        reference = np.array([3., 1., 2.])
        head = create_head().fit(reference)
        np.testing.assert_array_equal(reference, [3., 1., 2.])
        state = copy.deepcopy(head.to_dict())
        reference[:] = -999.
        for queries in ([50., -30.], [0., 0., 0.], []):
            head.score(queries)
        self.assertEqual(head.to_dict(), state)
        state['reference_logits'][0] = 555.
        self.assertEqual(head.to_dict()['reference_logits'], [1., 2., 3.])

    def test_swappable_heads_roundtrip_and_common_decision_contract(self):
        for identifier in ('empirical_tail', 'fixed_likelihood'):
            with self.subTest(identifier=identifier):
                head = create_head(identifier).fit(np.linspace(-2., 2., 99))
                original = head.score([-100., 0., 100.])
                restored = load_head(json.loads(json.dumps(head.to_dict(), allow_nan=False)))
                self.assertEqual(restored.reference_count, 99)
                for key in ('tail_probability', 'score'):
                    np.testing.assert_array_equal(original[key], restored.score([-100., 0., 100.])[key])
                decisions = original['score'] > head.get_threshold(.05)
                np.testing.assert_array_equal(decisions, [True, False, False])
                self.assertEqual(head.score([])['score'].shape, (0,))
                self.assertAlmostEqual(head.get_threshold(.05), 4.321928094887363)

    def test_fixed_head_sigmoid_and_numerical_limits(self):
        head = create_head('fixed_likelihood').fit([1.])
        scored = head.score([-1e300, -2., 0., 2., 1e300])
        np.testing.assert_allclose(scored['tail_probability'][1:4],
                                   1 / (1 + np.exp(-np.array([-2., 0., 2.]))))
        self.assertTrue(np.all(np.isfinite(scored['score'])))
        self.assertTrue(np.all(scored['tail_probability'] > 0))
        self.assertTrue(np.all(np.diff(scored['score']) <= 0))
        np.testing.assert_array_equal(scored['score'], -np.log2(scored['tail_probability']))

    def test_invalid_inputs_and_unfitted_use(self):
        for identifier in ('empirical_tail', 'fixed_likelihood'):
            with self.subTest(identifier=identifier):
                head = create_head(identifier)
                with self.assertRaisesRegex(ValueError, 'fitted'): head.score([0.])
                with self.assertRaisesRegex(ValueError, 'fitted'): head.to_dict()
                for bad in ([], [float('nan')], [float('inf')], [-float('inf')], [[1., 2.]], 1., ['bad']):
                    with self.assertRaises(ValueError): head.fit(bad)
                head.fit([0., 1.])
                for bad in ([float('nan')], [float('inf')], [[1.]], 1., ['bad']):
                    with self.assertRaises(ValueError): head.score(bad)
                for alpha in (0., -1., 1.01, float('nan'), float('inf'), True, None, [0.05]):
                    with self.assertRaises(ValueError): head.get_threshold(alpha)
                self.assertEqual(head.get_threshold(1.), 0.)

    def test_registry_and_saved_state_validation(self):
        self.assertEqual({item['id'] for item in manifest()['prediction_heads']},
                         {'empirical_tail', 'fixed_likelihood'})
        for identifier in ('os.system', '../arbitrary', None, {}, '__import__'):
            with self.assertRaises(ValueError): create_head(identifier)
            with self.assertRaises(ValueError): load_head({'version': 1, 'id': identifier})
        for identifier in ('empirical_tail', 'fixed_likelihood'):
            state = create_head(identifier).fit([0., 1.]).to_dict()
            for bad in ({**state, 'version': 2}, {**state, 'version': True},
                        {**state, 'python_module': 'os'}, {'id': identifier}):
                with self.assertRaises(ValueError): load_head(bad)
        for count in (0, -1, True, 1.5, '2'):
            with self.assertRaises(ValueError):
                load_head({'version': 1, 'id': 'fixed_likelihood', 'reference_count': count})
        for reference in ([], [float('nan')], [[1., 2.]]):
            with self.assertRaises(ValueError):
                load_head({'version': 1, 'id': 'empirical_tail', 'reference_logits': reference})
        with self.assertRaises(ValueError): load_head([])


if __name__ == '__main__':
    unittest.main()
