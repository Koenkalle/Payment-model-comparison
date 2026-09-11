"""Exercise the actual loopback HTTP boundary independently of Torch weights."""
import concurrent.futures
import copy
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.comparison_service import ComparisonService, MAX_BODY_BYTES, make_server


DOCUMENT = {
    'schema': 'payment-events/v1',
    'accounts': [{'id': 0}, {'id': 1}],
    'events': [{'id': 'P1', 'kind': 'payment', 't': 1, 'u': 0, 'v': 1, 'amount': 25}],
    'truth': {'P1': False},
}


class Scorer:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.delay = 0
        self.failure = None

    def compare(self, document, options, cancelled=None):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                for _ in range(10):
                    if cancelled and cancelled():
                        raise InterruptedError('Comparison superseded by newer settings.')
                    time.sleep(self.delay / 10)
            if self.failure is not None:
                failure, self.failure = self.failure, None
                raise failure
            return {'schema': 'native-fraud-comparison/v2', 'dataset': document, 'options': options,
                    'predictions': [{'id': document['events'][0]['id'], 'logit': float(self.calls)}]}
        finally:
            self.active -= 1


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'index.html').write_text('<!doctype html><title>Comparison</title>')
        (self.root / '.git').mkdir()
        (self.root / '.git' / 'config').write_text('not public')
        self.scorer = Scorer()
        self.service = ComparisonService(scorer_factory=lambda *_: self.scorer,
                                         supervised_artifact=None, cache_entries=2)
        self.server = make_server(self.service, port=0, root=self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def request(self, method='GET', path='/api/models', payload=None, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(payload).encode()
            headers.setdefault('Content-Type', 'application/json')
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        content = response.read()
        result = (response.status, dict(response.getheaders()), content)
        connection.close()
        return result

    def payload(self, **options):
        return {'model_id': 'dyg_tami_native', 'dataset': copy.deepcopy(DOCUMENT), 'options': options}

    def test_discovery_static_and_same_origin_json(self):
        status, _, body = self.request()
        model = json.loads(body)['models'][0]
        self.assertEqual(status, 200)
        self.assertTrue(model['available'])
        self.assertEqual(model['capabilities']['modes'], ['shadow', 'enforce'])
        self.assertEqual(model['capabilities']['training_modes'], ['unsupervised'])
        self.assertEqual(set(model['capabilities']['prediction_heads']), {'empirical_tail', 'fixed_likelihood'})
        status, headers, body = self.request(path='/')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'text/html')
        self.assertIn(b'Comparison', body)
        self.assertEqual(self.request('HEAD', '/')[2], b'')
        origin = f'http://127.0.0.1:{self.server.server_port}'
        status, headers, body = self.request('POST', '/api/compare', self.payload(), headers={'Origin': origin})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result['dataset'], DOCUMENT)
        self.assertEqual(result['options']['predictionHead'], 'empirical_tail')
        self.assertNotIn('Access-Control-Allow-Origin', headers)

    def test_reject_bad_model_settings_and_documents_without_inference(self):
        variants = [
            [], {'model_id': 'unknown', 'dataset': DOCUMENT},
            {'model_id': 'dyg_tami_native', 'dataset': '/tmp/private.json'},
            self.payload(trainingMode='unknown'), self.payload(trainingMode='supervised', predictionHead='empirical_tail'),
            self.payload(mode='invalid'),
            self.payload(decisionPolicy='unknown'), self.payload(predictionHead='unknown'),
            self.payload(alpha=0), self.payload(alpha=True), self.payload(warmup=1.5),
            self.payload(falseBlockCost=-1), self.payload(eta=2), self.payload(artifact='/tmp/file'),
        ]
        for payload in variants:
            with self.subTest(payload=payload):
                status, _, body = self.request('POST', '/api/compare', payload)
                self.assertEqual(status, 400, body)
                self.assertIn('error', json.loads(body))
        malformed = self.payload()
        malformed['dataset']['events'] = [7]
        self.assertEqual(self.request('POST', '/api/compare', malformed)[0], 400)
        self.assertEqual(self.scorer.calls, 0)

    def test_cache_includes_stream_truth_and_settings_and_is_bounded(self):
        payload = self.payload(mode='shadow')
        first = self.service.compare(payload)
        # Normalized defaults give the same cache entry and returned objects
        # cannot mutate cached JSON.
        first['predictions'][0]['logit'] = 999
        self.assertEqual(self.service.compare(self.payload(mode='shadow', alpha=.02))['predictions'][0]['logit'], 1)
        self.assertEqual(self.scorer.calls, 1)
        changed = copy.deepcopy(payload)
        changed['dataset']['events'][0]['amount'] += 1
        self.service.compare(changed)
        changed['dataset']['truth']['P1'] = True
        self.service.compare(changed)
        self.assertEqual(self.scorer.calls, 3)
        self.service.compare(payload)  # First entry was evicted.
        self.assertEqual(self.scorer.calls, 4)
        for options in ({'mode': 'enforce'}, {'predictionHead': 'fixed_likelihood'}, {'alpha': .1}):
            self.service.compare(self.payload(**options))
        self.assertEqual(self.scorer.calls, 7)
        self.assertEqual(len(self.service._cache), 2)
        self.assertEqual(self.service._cache_bytes, sum(map(len, self.service._cache.values())))
        no_cache = ComparisonService(scorer_factory=lambda *_: self.scorer, max_cache_bytes=1)
        no_cache.compare(payload)
        no_cache.compare(payload)
        self.assertEqual(len(no_cache._cache), 0)
        self.assertEqual(self.scorer.calls, 9)

    def test_parallel_requests_serialize_torch_work_and_reuse_duplicates(self):
        self.scorer.delay = .03
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.request('POST', '/api/compare', self.payload()), range(4)))
        self.assertTrue(all(status == 200 for status, _, _ in results))
        self.assertEqual(self.scorer.calls, 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda alpha: self.service.compare(self.payload(alpha=alpha)), [.03, .04, .05, .06]))
        self.assertEqual(len(results), 4)
        self.assertEqual(self.scorer.peak, 1)

    def test_http_rejects_cross_origin_hidden_paths_large_and_non_json_requests(self):
        for headers in ({'Origin': 'https://example.com'}, {'Host': 'example.com:' + str(self.server.server_port)},
                        {'Sec-Fetch-Site': 'cross-site'}):
            self.assertEqual(self.request('POST', '/api/compare', self.payload(), headers=headers)[0], 403)
        for path in ('/.git/config', '/../secret', '/%2e%2e/secret', '/api/missing'):
            self.assertEqual(self.request(path=path)[0], 404)
        outside = self.root.parent / (self.root.name + '-outside.txt')
        try:
            outside.write_text('not served')
            (self.root / 'link.txt').symlink_to(outside)
            self.assertEqual(self.request(path='/link.txt')[0], 404)
        finally:
            outside.unlink(missing_ok=True)
        self.assertEqual(self.request('POST', '/api/compare', body=b'{}')[0], 415)
        headers = {'Content-Type': 'application/json'}
        self.assertEqual(self.request('POST', '/api/compare', body=b'{', headers=headers)[0], 400)
        self.assertEqual(self.request('POST', '/api/compare', body=b'{"alpha":NaN}', headers=headers)[0], 400)
        headers['Content-Length'] = str(MAX_BODY_BYTES + 1)
        self.assertEqual(self.request('POST', '/api/compare', body=b'{}', headers=headers)[0], 413)
        with self.assertRaisesRegex(ValueError, 'loopback'):
            make_server(self.service, host='0.0.0.0', port=0)

    def test_unavailable_model_and_failed_requests_can_recover(self):
        def unavailable(*_):
            raise ImportError('Install requirements-temporal.txt')
        self.service._factory = unavailable
        status, _, body = self.request()
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)['models'][0]['available'])
        self.assertEqual(self.request('POST', '/api/compare', self.payload())[0], 503)
        self.service._factory = lambda *_: self.scorer
        self.scorer.failure = ValueError('Dataset cannot be scored')
        self.assertEqual(self.request('POST', '/api/compare', self.payload())[0], 400)
        self.assertEqual(self.request('POST', '/api/compare', self.payload())[0], 200)
        self.assertEqual(self.scorer.calls, 2)
        self.assertTrue(json.loads(self.request()[2])['models'][0]['available'])

    def test_new_settings_cancel_running_and_queued_work(self):
        self.scorer.delay = .5
        first = {'X-Comparison-Session': 'page-1', 'X-Comparison-Revision': '1'}
        second = {**first, 'X-Comparison-Revision': '2'}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            older = pool.submit(self.request, 'POST', '/api/compare', self.payload(alpha=.1), headers=first)
            deadline = time.monotonic() + 2
            while not self.scorer.active and time.monotonic() < deadline:
                time.sleep(.005)
            newer = pool.submit(self.request, 'POST', '/api/compare', self.payload(alpha=.2), headers=second)
            self.assertEqual(older.result()[0], 409)
            self.assertEqual(newer.result()[0], 200)
        self.assertEqual(len(self.service._cache), 1)
        self.assertEqual(self.request('POST', '/api/compare', self.payload(alpha=.1), headers=first)[0], 409)
        self.assertEqual(self.scorer.calls, 2)
        self.assertEqual(self.request('POST', '/api/compare', self.payload(), headers={
            'X-Comparison-Session': 'page-1'})[0], 400)
        for i in range(40):
            self.service.register_request('page-' + str(i), '1')
        self.assertEqual(len(self.service._sessions), 32)
        self.assertTrue(self.service.register_request('latest', '1')() is False)

    def test_browser_abort_cancels_work_without_requiring_a_following_request(self):
        self.scorer.delay = 1
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request('POST', '/api/compare', body=json.dumps(self.payload()),
                           headers={'Content-Type': 'application/json'})
        deadline = time.monotonic() + 2
        while not self.scorer.active and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.scorer.active, 1)
        connection.close()
        deadline = time.monotonic() + .5
        while self.scorer.active and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.scorer.active, 0)
        self.assertEqual(len(self.service._cache), 0)

    def test_capabilities_follow_actual_configured_artifact(self):
        self.scorer.describe = lambda: {
            'checkpoint_id': 'test-checkpoint', 'supported_modes': ['shadow', 'enforce'],
            'supported_policies': ['shared', 'manual'], 'training_modes': ['unsupervised'],
            'prediction_heads': [{'id': 'empirical_tail', 'label': 'Historical rank'}],
        }
        description = self.service.models()['models'][0]
        self.assertEqual(description['checkpoint_id'], 'test-checkpoint')
        self.assertEqual(description['capabilities']['decision_policies'], ['shared', 'manual'])
        self.assertEqual(description['capabilities']['prediction_heads'], ['empirical_tail'])

    def test_supervised_mode_routes_to_separate_artifact_and_head(self):
        supervised = Scorer()
        supervised.describe = lambda: {
            'checkpoint_id': 'fraud-checkpoint', 'supported_policies': ['shared', 'manual', 'tuned', 'auto'],
            'training_modes': ['supervised'], 'prediction_heads': [{'id': 'fraud_linear'}],
            'default_head': 'fraud_linear',
        }
        self.service.supervised_artifact = self.root / 'fraud-artifact'
        self.service._supervised_factory = lambda *_: supervised
        description = self.service.models()['models'][0]
        self.assertEqual(description['capabilities']['training_modes'], ['unsupervised', 'supervised'])
        modes = description['capabilities']['training_mode_capabilities']
        self.assertEqual(modes['supervised']['checkpoint_id'], 'fraud-checkpoint')
        self.assertEqual(modes['supervised']['default_head'], 'fraud_linear')
        self.assertTrue(modes['unsupervised']['available'])
        status, _, body = self.request('POST', '/api/compare', self.payload(trainingMode='supervised'))
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)['options']['predictionHead'], 'fraud_linear')
        self.assertEqual(json.loads(body)['options']['manualTau'], 1)
        self.assertEqual(supervised.calls, 1)
        self.assertEqual(self.scorer.calls, 0)
        self.assertEqual(self.request('POST', '/api/compare', self.payload())[0], 200)
        self.assertEqual(self.scorer.calls, 1)
        # Each mode reuses only its own frozen checkpoint and prediction cache.
        self.service.compare(self.payload(trainingMode='supervised'))
        self.assertEqual(supervised.calls, 1)
        self.assertEqual(self.request('POST', '/api/compare', self.payload(predictionHead='fraud_linear'))[0], 400)

    def test_missing_mode_is_isolated_and_never_falls_back(self):
        description = self.service.models()['models'][0]
        self.assertTrue(description['available'])
        self.assertFalse(description['capabilities']['training_mode_capabilities']['supervised']['available'])
        self.assertEqual(self.request('POST', '/api/compare', self.payload(trainingMode='supervised'))[0], 503)
        self.assertEqual(self.scorer.calls, 0)
        supervised = Scorer()
        supervised.describe = lambda: {'training_modes': ['supervised'],
                                       'prediction_heads': [{'id': 'fraud_linear'}]}
        self.service.supervised_artifact = self.root / 'fraud-artifact'
        self.service._supervised_factory = lambda *_: supervised
        self.service._scorers.clear()
        def missing(*_):
            raise OSError('Missing link checkpoint')
        self.service._factory = missing
        description = self.service.models()['models'][0]
        self.assertTrue(description['available'])
        self.assertEqual(description['capabilities']['training_modes'], ['supervised'])
        self.assertEqual(self.request('POST', '/api/compare', self.payload())[0], 503)
        self.assertEqual(self.request('POST', '/api/compare', self.payload(trainingMode='supervised'))[0], 200)


if __name__ == '__main__':
    unittest.main()
