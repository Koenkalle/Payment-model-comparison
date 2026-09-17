"""Parameter help is a complete, serializable part of the public model catalog."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.pipeline_training import MODEL_SETTINGS, TrainingService, _field, _validate_parameters
from models.implementations import tabfm


class ParameterInfoTests(unittest.TestCase):
    def catalog(self):
        # Catalog serialization needs no worker, filesystem, model fit or download.
        service = TrainingService.__new__(TrainingService)
        service._dependencies = {'sklearn': None, 'torch': None, 'xgboost': None}
        with patch('framework.pipeline_training.importlib.import_module',
                   return_value=SimpleNamespace(availability_error=lambda: None)):
            return service.models()

    def test_every_public_parameter_carries_extended_help_and_live_constraints(self):
        models = self.catalog()
        self.assertEqual({model['id'] for model in models}, set(MODEL_SETTINGS))
        json.dumps(models, allow_nan=False)
        for model in models:
            expected = MODEL_SETTINGS[model['id']]['parameters']
            self.assertEqual(model['parameters'], expected)
            self.assertEqual(len({field['name'] for field in expected}), len(expected))
            for field in model['parameters']:
                with self.subTest(model=model['id'], field=field['name']):
                    info = field['info']
                    self.assertEqual(info['title'], field['label'])
                    self.assertIsInstance(info['description'], str)
                    self.assertGreater(len(info['description']), 50)
                    self.assertGreaterEqual(len(info['sections']), 2)
                    for section in info['sections']:
                        self.assertTrue(section['title'])
                        self.assertIsInstance(section['text'], str)
                        self.assertGreater(len(section['text']), 50)
                    # Defaults and constraints have one owner: the field used
                    # by validation. No second copy of them lives in help data.
                    self.assertFalse({'default', 'min', 'max', 'options'} & set(info))
            self.assertEqual(_validate_parameters(model['id'], {}),
                             {field['name']: field['default'] for field in expected})

    def test_future_fields_must_supply_help_and_catalog_results_are_isolated(self):
        with self.assertRaises(TypeError):
            _field('undocumented', 'Undocumented', 1, 0, 2)
        before = copy.deepcopy(MODEL_SETTINGS)
        first = self.catalog()
        first[0]['parameters'][0]['info']['sections'][0]['text'] = 'client mutation'
        first[0]['parameters'][0]['info']['description'] = 'client mutation'
        self.assertEqual(MODEL_SETTINGS, before)
        self.assertNotEqual(self.catalog()[0]['parameters'][0], first[0]['parameters'][0])

    def test_tabfm_displayed_limits_and_defaults_match_native_context_adapter(self):
        fields = {field['name']: field for field in MODEL_SETTINGS['tabfm']['parameters']}
        self.assertEqual(set(fields) - {'decision_threshold'}, set(tabfm.DEFAULT_PARAMETERS))
        for name, default in tabfm.DEFAULT_PARAMETERS.items():
            with self.subTest(parameter=name):
                self.assertEqual(fields[name]['default'], default)
                self.assertEqual((fields[name]['min'], fields[name]['max']), tabfm.PARAMETER_LIMITS[name])
                for value in tabfm.PARAMETER_LIMITS[name]:
                    self.assertEqual(_validate_parameters('tabfm', {name: value})[name], value)
        self.assertEqual(MODEL_SETTINGS['tabfm']['max_features'], tabfm.MAX_FEATURES)

    def test_displayed_graph_architectures_satisfy_explained_attention_constraint(self):
        fields = {field['name']: field for field in MODEL_SETTINGS['dyg_tami_native']['parameters']}
        # The help explains four concatenated channels and an evenly divided
        # attention width. Every combination the UI offers must satisfy it.
        for width in fields['channel_embedding_dim']['options']:
            for heads in fields['num_heads']['options']:
                with self.subTest(width=width, heads=heads):
                    self.assertEqual((4 * width) % heads, 0)


if __name__ == '__main__':
    unittest.main()
