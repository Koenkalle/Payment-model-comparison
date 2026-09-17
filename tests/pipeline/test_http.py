"""Exercise the pipeline through the same loopback JSON boundary as the pages."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.comparison_service import make_server
from framework.pipeline_service import PipelineService


class PipelineHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pipeline = PipelineService(Path(self.temp.name) / 'saved')
        self.server = make_server(SimpleNamespace(models=lambda: {'models': []}), port=0,
                                  pipeline_service=self.pipeline)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.pipeline.close()
        self.temp.cleanup()

    def request(self, path, payload=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=30)
        request_headers = {'Content-Type': 'application/json', **(headers or {})}
        connection.request('GET' if payload is None else 'POST', '/api/pipeline' + path,
                           body=None if payload is None else json.dumps(payload), headers=request_headers)
        response = connection.getresponse()
        status, result = response.status, json.loads(response.read())
        connection.close()
        return status, result

    def test_generate_train_compare_and_reopen_over_http(self):
        status, catalog = self.request('/datasets')
        self.assertEqual(status, 200)
        self.assertTrue(catalog['datasets'])
        status, created = self.request('/datasets/generate', {
            'generator': 'handbook_generator', 'name': 'HTTP benchmark',
            'parameters': {'transactions': 160, 'fraud_rate': .3, 'seed': 42}})
        self.assertEqual(status, 201, created)
        identifier = created['dataset']['id']
        status, detail = self.request('/datasets/' + identifier)
        self.assertEqual(status, 200)
        self.assertEqual(detail['dataset']['rows'], 160)
        self.assertTrue(detail['sample']['rows'])
        self.assertNotIn(str(self.temp.name), json.dumps(detail))
        status, models = self.request('/models?dataset_id=' + identifier)
        self.assertEqual(status, 200)
        self.assertTrue(next(row for row in models['models'] if row['id'] == 'logistic_regression')['available'])
        status, submitted = self.request('/train', {'dataset_id': identifier,
            'model_id': 'logistic_regression', 'parameters': {'C': .25},
            'split': {'train': .6, 'validation': .2}})
        self.assertEqual(status, 202, submitted)
        job_id = submitted['job']['id']
        self.pipeline.training._futures[job_id].result(timeout=40)
        status, finished = self.request('/jobs/' + job_id)
        self.assertEqual(finished['job']['status'], 'succeeded', finished)
        run = finished['job']['result']['run']
        status, ready = self.request('/runs')
        self.assertEqual(ready['runs'][0]['id'], run['id'])
        status, comparison = self.request('/compare', {'dataset_id': identifier,
            'run_ids': [run['id']], 'partition': 'test'})
        self.assertEqual(status, 202, comparison)
        comparison_id = comparison['job']['id']
        self.pipeline.training._futures[comparison_id].result(timeout=40)
        _, result = self.request('/jobs/' + comparison_id)
        self.assertEqual(result['job']['status'], 'succeeded', result)
        report = result['job']['result']
        self.assertEqual(report['row_count'], run['partition_counts']['test'])
        self.assertEqual(report['training_rows_in_evaluation'], 0)
        self.assertTrue(report['rows'])
        _, listed = self.request('/jobs')
        summary = next(job for job in listed['jobs'] if job['id'] == comparison_id)
        self.assertNotIn('rows', summary['result'])

    def test_untrusted_or_invalid_requests_do_not_create_datasets(self):
        before = len(self.pipeline.store.catalog()['datasets'])
        valid = {'generator': 'handbook_generator'}
        status, _ = self.request('/datasets/generate', valid, {'Origin': 'http://evil.example'})
        self.assertEqual(status, 403)
        for payload in ({'generator': 'handbook_generator', 'path': '/etc/passwd'}, [],
                        {'generator': 'handbook_generator', 'parameters': {'transactions': 10**9}}):
            status, result = self.request('/datasets/generate', payload)
            self.assertEqual(status, 400, result)
        self.assertEqual(len(self.pipeline.store.catalog()['datasets']), before)
        self.assertEqual(self.request('/unknown')[0], 404)
        self.assertEqual(self.request('/datasets/..%2F..%2Fetc%2Fpasswd')[0], 404)

    def test_feature_catalog_and_immutable_selection_over_http(self):
        _, catalog = self.request('/datasets')
        parent = catalog['datasets'][0]
        path = '/datasets/' + parent['id'] + '/features'
        status, features = self.request(path)
        self.assertEqual(status, 200, features)
        self.assertEqual(features['enabled_features'], parent['feature_names'])
        self.assertTrue(features['features'][0]['statistics'])
        status, result = self.request(path, {'features': ['log_amount', 'sender_out_count'], 'name': 'Shared features'})
        self.assertEqual(status, 201, result)
        variant = result['dataset']
        self.assertNotEqual(variant['id'], parent['id'])
        _, detail = self.request('/datasets/' + variant['id'])
        self.assertEqual(detail['sample']['columns'], ['log_amount', 'sender_out_count'])
        _, saved = self.request('/datasets/' + variant['id'] + '/features')
        self.assertEqual(saved['enabled_features'], ['log_amount', 'sender_out_count'])
        self.assertNotIn(str(self.temp.name), json.dumps(saved))
        self.assertEqual(self.request(path, {'features': ['TX_FRAUD']})[0], 400)
        self.assertEqual(self.request(path, {'features': ['amount'], 'path': '/etc/passwd'})[0], 400)
        self.assertEqual(self.request(path, {'features': ['amount']}, {'Origin': 'http://evil.example'})[0], 403)
        _, original = self.request('/datasets/' + parent['id'])
        self.assertEqual(original['dataset'], parent)


if __name__ == '__main__':
    unittest.main()
