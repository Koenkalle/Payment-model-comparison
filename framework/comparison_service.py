"""Local HTTP boundary for native models used by the comparison page.

Model code stays in ``native_comparison``. This module validates requests,
serializes stateful Torch work, and caches complete results for an exact dataset
and settings. HTTP clients can submit data, never server-side paths or code.
"""
from collections import OrderedDict
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import mimetypes
from pathlib import Path
import re
import select
import socket
import threading
import traceback
from urllib.parse import unquote, urlsplit

from prediction_heads.registry import manifest as prediction_head_manifest


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'dyg_tami_native'
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_ACCOUNTS = 256
MAX_EVENTS = 20000
DEFAULT_ARTIFACT = ROOT / 'models' / 'native-dyg-tami'
DEFAULT_SUPERVISED_ARTIFACT = ROOT / 'models' / 'native-dyg-tami-fraud'
MODE_CAPABILITIES = {
    'unsupervised': {'prediction_heads': [entry['id'] for entry in prediction_head_manifest()['prediction_heads']],
                     'default_head': 'empirical_tail'},
    'supervised': {'prediction_heads': [entry['id'] for entry in prediction_head_manifest(kind='transaction-representation')['prediction_heads']],
                   'default_head': 'fraud_linear'},
}
CAPABILITIES = {
    'modes': ['shadow', 'enforce'],
    'training_modes': list(MODE_CAPABILITIES),
    'prediction_heads': [identifier for entry in MODE_CAPABILITIES.values() for identifier in entry['prediction_heads']],
    'decision_policies': ['shared', 'manual', 'tuned', 'auto'],
    'max_accounts': MAX_ACCOUNTS,
    'max_events': MAX_EVENTS,
}
OPTION_DEFAULTS = {
    'mode': 'enforce', 'trainingMode': 'unsupervised',
    'decisionPolicy': 'shared', 'alpha': 0.02, 'warmup': 128,
    'manualTau': 10.0, 'falseBlockCost': 1.0, 'missedFraudCost': 20.0,
    'objective': 'f1', 'predictionHead': 'empirical_tail', 'eta': 0.025,
}


class ModelUnavailable(RuntimeError):
    """The configured local native model cannot currently be loaded."""


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_request(payload):
    """Validate the public request without importing optional model libraries."""
    if not isinstance(payload, dict) or set(payload) - {'model_id', 'dataset', 'options'}:
        raise ValueError('Expected model_id, dataset and optional options.')
    if payload.get('model_id') != MODEL_ID:
        raise ValueError('Unsupported comparison model: ' + str(payload.get('model_id')))
    document = payload.get('dataset')
    if not isinstance(document, dict):
        raise ValueError('dataset must be a payment-events/v1 object, not a file path.')
    accounts, events = document.get('accounts'), document.get('events')
    if not isinstance(accounts, list) or not accounts or len(accounts) > MAX_ACCOUNTS:
        raise ValueError(f'Dataset requires 1–{MAX_ACCOUNTS} accounts.')
    if not isinstance(events, list) or not events or len(events) > MAX_EVENTS:
        raise ValueError(f'Dataset requires 1–{MAX_EVENTS} events.')
    if not all(isinstance(row, dict) for row in accounts + events):
        raise ValueError('Accounts and events must contain objects.')
    options = payload.get('options', {})
    if not isinstance(options, dict) or set(options) - set(OPTION_DEFAULTS):
        raise ValueError('Unknown comparison setting.')
    supplied_options = options
    options = {**OPTION_DEFAULTS, **options}
    if not isinstance(options['trainingMode'], str):
        raise ValueError('trainingMode must be a registered mode name.')
    mode_capabilities = MODE_CAPABILITIES.get(options['trainingMode'], {})
    if 'predictionHead' not in supplied_options and mode_capabilities:
        options['predictionHead'] = mode_capabilities['default_head']
    if options['trainingMode'] == 'supervised' and 'manualTau' not in supplied_options:
        options['manualTau'] = 1.0
    choices = {
        'mode': CAPABILITIES['modes'],
        'trainingMode': CAPABILITIES['training_modes'],
        'decisionPolicy': CAPABILITIES['decision_policies'],
        'predictionHead': mode_capabilities.get('prediction_heads', []),
        'objective': ['f1', 'f2', 'balanced_accuracy'],
    }
    for key, allowed in choices.items():
        if options[key] not in allowed:
            raise ValueError(f'Unsupported {key}: {options[key]}. Supported: ' + ', '.join(allowed))
    if not _finite_number(options['alpha']) or not 0 < options['alpha'] < 1:
        raise ValueError('alpha must be strictly between zero and one.')
    if type(options['warmup']) is not int or not 1 <= options['warmup'] <= MAX_EVENTS:
        raise ValueError(f'warmup must be an integer between 1 and {MAX_EVENTS}.')
    if not _finite_number(options['manualTau']):
        raise ValueError('manualTau must be finite.')
    for key in ('falseBlockCost', 'missedFraudCost'):
        if not _finite_number(options[key]) or options[key] <= 0:
            raise ValueError(key + ' must be finite and positive.')
    if not _finite_number(options['eta']) or not 0 <= options['eta'] <= 1:
        raise ValueError('eta must be between zero and one.')
    # Reject NaN/Infinity anywhere, including metadata or truth; both would
    # otherwise have nonstandard JSON encodings and unstable cache identities.
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Dataset must contain finite JSON values.') from error
    return document, options


def _native_factory(artifact, calibration_config):
    from .native_comparison import NativeComparison
    return NativeComparison(artifact, calibration_config)


class ComparisonService:
    """Isolated training-mode artifacts sharing serialized CPU inference."""

    def __init__(self, artifact=DEFAULT_ARTIFACT, calibration_config=None, *,
                 supervised_artifact=DEFAULT_SUPERVISED_ARTIFACT,
                 scorer_factory=None, supervised_scorer_factory=None,
                 cache_entries=4, max_cache_bytes=32 * 1024 * 1024):
        self.artifact = Path(artifact).resolve()
        self.calibration_config = Path(calibration_config or self.artifact / 'calibration.config.json').resolve()
        self.supervised_artifact = Path(supervised_artifact).resolve() if supervised_artifact is not None else None
        if type(cache_entries) is not int or cache_entries < 0 or max_cache_bytes < 0:
            raise ValueError('Cache limits must be nonnegative.')
        self._factory = scorer_factory or _native_factory
        self._supervised_factory = supervised_scorer_factory or _native_factory
        self._scorers = {}
        self._descriptions = {}
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_entries = cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._revision_lock = threading.Lock()
        self._sessions = OrderedDict()

    def _load_locked(self, training_mode='unsupervised'):
        if training_mode not in self._scorers:
            try:
                if training_mode == 'unsupervised':
                    scorer = self._factory(self.artifact, self.calibration_config)
                else:
                    if self.supervised_artifact is None:
                        raise ValueError('No supervised artifact is configured.')
                    scorer = self._supervised_factory(self.supervised_artifact, None)
                description = scorer.describe() if hasattr(scorer, 'describe') else {}
                if training_mode not in description.get('training_modes', [training_mode]):
                    raise ValueError('The configured artifact does not support ' + training_mode + ' training.')
                # Snapshot identity from the verified, loaded scorer. Do not read
                # files again: training may replace them while this process serves
                # the previous weights from memory.
                description = copy.deepcopy(description)
                if description.get('artifact'):
                    artifact = description['artifact']
                    default = DEFAULT_ARTIFACT if training_mode == 'unsupervised' else DEFAULT_SUPERVISED_ARTIFACT
                    artifact['source'] = 'default' if Path(artifact['path']).resolve() == default.resolve() else 'custom'
                self._scorers[training_mode] = scorer
                self._descriptions[training_mode] = description
            except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                self._scorers.pop(training_mode, None)
                self._descriptions.pop(training_mode, None)
                message = str(error)
                if isinstance(error, ImportError):
                    message += '; install requirements-temporal.txt in the environment running serve.py.'
                raise ModelUnavailable(message) from error
        return self._scorers[training_mode]

    def models(self):
        mode_capabilities = {}
        with self._lock:
            for mode, defaults in MODE_CAPABILITIES.items():
                error = None
                try:
                    self._load_locked(mode)
                except ModelUnavailable as unavailable:
                    error = str(unavailable)
                description = self._descriptions.get(mode, {})
                mode_capabilities[mode] = {
                    'available': error is None, 'error': error,
                    'checkpoint_id': description.get('checkpoint_id'),
                    'artifact': copy.deepcopy(description.get('artifact')),
                    'feature_contract': description.get('feature_contract'),
                    'modes': list(description.get('supported_modes', CAPABILITIES['modes'])),
                    'prediction_heads': [head['id'] for head in description.get('prediction_heads', [])]
                    if 'prediction_heads' in description else list(defaults['prediction_heads']),
                    'decision_policies': list(description.get('supported_policies', CAPABILITIES['decision_policies'])),
                    'default_head': description.get('default_head', defaults['default_head']),
                }
        capabilities = {key: list(value) if isinstance(value, list) else value
                        for key, value in CAPABILITIES.items()}
        available = {mode: entry for mode, entry in mode_capabilities.items() if entry['available']}
        capabilities['training_mode_capabilities'] = mode_capabilities
        capabilities['training_modes'] = list(available)
        for key in ('modes', 'prediction_heads', 'decision_policies'):
            capabilities[key] = list(dict.fromkeys(value for entry in available.values() for value in entry[key]))
        primary = next(iter(available.values()), {})
        return {'version': 1, 'models': [{
            'id': MODEL_ID, 'label': 'DyGFormer + TAMI (PyTorch)',
            'available': bool(available),
            'error': None if available else '; '.join(mode + ': ' + entry['error'] for mode, entry in mode_capabilities.items()),
            'checkpoint_id': primary.get('checkpoint_id'),
            'capabilities': capabilities,
        }]}

    def register_request(self, session=None, revision=None):
        """Return a cheap cancellation predicate for this page's settings."""
        if session is None and revision is None:
            return lambda: False
        if (not isinstance(session, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session)
                or not isinstance(revision, str) or not revision.isdecimal()
                or len(revision) > 16 or int(revision) > 2 ** 53 - 1):
            raise ValueError('Comparison session and integer revision headers must be supplied together.')
        revision = int(revision)
        with self._revision_lock:
            self._sessions[session] = max(revision, self._sessions.get(session, -1))
            self._sessions.move_to_end(session)
            while len(self._sessions) > 32:
                self._sessions.popitem(last=False)
        def cancelled():
            with self._revision_lock:
                return self._sessions.get(session) != revision
        return cancelled

    def compare_bytes(self, payload, cancelled=None):
        document, options = validate_request(payload)
        cancelled = cancelled or (lambda: False)
        def check_cancelled():
            if cancelled():
                raise InterruptedError('Comparison superseded by newer settings.')
        check_cancelled()
        with self._lock:
            check_cancelled()
            scorer = self._load_locked(options['trainingMode'])
            description = self._descriptions.get(options['trainingMode'], {})
            if 'predictionHead' not in payload.get('options', {}) and description.get('default_head'):
                options['predictionHead'] = description['default_head']
            if options['mode'] not in description.get('supported_modes', CAPABILITIES['modes']):
                raise ValueError('The selected history mode is unavailable in the configured artifact.')
            if options['predictionHead'] not in [head['id'] for head in description.get('prediction_heads', [{'id': options['predictionHead']}])]:
                raise ValueError('The selected head is unavailable in the configured artifact.')
            if options['decisionPolicy'] not in description.get('supported_policies', CAPABILITIES['decision_policies']):
                raise ValueError('The selected policy is unavailable for this artifact; historical validation needs both known classes.')
            canonical = json.dumps({'dataset': document, 'options': options,
                                    'checkpoint': description.get('checkpoint_id')},
                                   sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
            key = hashlib.sha256(canonical).digest()
            check_cancelled()
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            result = scorer.compare(document, options, cancelled=cancelled)
            check_cancelled()
            if description.get('artifact'):
                if result.get('model', {}).get('checkpoint_id') != description.get('checkpoint_id'):
                    raise ValueError('Prediction checkpoint differs from the loaded model identity.')
                result['model']['artifact'] = copy.deepcopy(description['artifact'])
            serialized = json.dumps(result, separators=(',', ':'), allow_nan=False).encode('utf-8')
            if self._cache_entries and len(serialized) <= self._max_cache_bytes:
                self._cache[key] = serialized
                self._cache_bytes += len(serialized)
                while len(self._cache) > self._cache_entries or self._cache_bytes > self._max_cache_bytes:
                    _, removed = self._cache.popitem(last=False)
                    self._cache_bytes -= len(removed)
            return serialized

    def compare(self, payload, cancelled=None):
        return json.loads(self.compare_bytes(payload, cancelled))


def _loopback(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ComparisonHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(service, host='127.0.0.1', port=8000, root=ROOT, *, dataset_service=None,
                pipeline_service=None):
    """Serve repo pages and the same-origin API on a loopback address only."""
    if not _loopback(host):
        raise ValueError('The comparison service only binds to a loopback address.')
    root = Path(root).resolve()
    if dataset_service is None:
        from .dataset_service import DatasetService
        dataset_service = DatasetService()

    class Handler(BaseHTTPRequestHandler):
        server_version = 'PaymentComparison/1'

        def setup(self):
            super().setup()
            self.connection.settimeout(60)

        def _trusted_request(self):
            try:
                authority = urlsplit('http://' + self.headers.get('Host', ''))
                if not _loopback(authority.hostname) or authority.port != self.server.server_port:
                    return False
                origin = self.headers.get('Origin')
                if origin:
                    source = urlsplit(origin)
                    if source.scheme != 'http' or source.netloc != authority.netloc:
                        return False
                return self.headers.get('Sec-Fetch-Site') != 'cross-site'
            except ValueError:
                return False

        def _send(self, status, body, content_type='application/json; charset=utf-8', head=False):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A newer setting may have cancelled the browser request.

        def _error(self, status, message):
            self._send(status, json.dumps({'error': message}).encode('utf-8'))

        def _get(self, head=False):
            if not self._trusted_request():
                self._error(403, 'This service accepts same-origin loopback requests only.')
                return
            try:
                pathname = unquote(urlsplit(self.path).path)
                if pathname.startswith('/api/pipeline/'):
                    if pipeline_service is None:
                        self._error(503, 'The preparation pipeline is unavailable. Start the app with python serve.py.')
                        return
                    try:
                        result = pipeline_service.get(pathname, urlsplit(self.path).query)
                        self._send(200, json.dumps(result, allow_nan=False).encode('utf-8'), head=head)
                    except (FileNotFoundError, KeyError) as error:
                        self._error(404, str(error))
                    except (ValueError, TypeError) as error:
                        self._error(400, str(error))
                    except Exception:
                        traceback.print_exc()
                        self._error(500, 'Pipeline lookup failed. Check the server output and retry.')
                    return
                if pathname == '/api/datasets':
                    self._send(200, json.dumps(dataset_service.catalog(), allow_nan=False).encode('utf-8'), head=head)
                    return
                if pathname == '/api/models':
                    try:
                        self._send(200, json.dumps(service.models(), allow_nan=False).encode('utf-8'), head=head)
                    except Exception:
                        traceback.print_exc()
                        self._error(500, 'Model discovery failed. Check the server output and retry.')
                    return
                if pathname.startswith('/api/'):
                    self._error(404, 'Unknown API endpoint.')
                    return
                relative = Path(pathname.lstrip('/'))
                if any(part.startswith('.') for part in relative.parts):
                    self._error(404, 'File not found.')
                    return
                path = (root / relative).resolve()
                if path.is_dir():
                    path = (path / 'index.html').resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    self._error(404, 'File not found.')
                    return
                self._send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream', head)
            except (OSError, ValueError):
                self._error(404, 'File not found.')

        def do_GET(self):
            self._get()

        def do_HEAD(self):
            self._get(head=True)

        def do_POST(self):
            if not self._trusted_request():
                self._error(403, 'This service accepts same-origin loopback requests only.')
                return
            endpoint = urlsplit(self.path).path
            graph_query = re.fullmatch(r'/api/pipeline/datasets/ds-[0-9a-f]{32}/graph/query', endpoint)
            feature_selection = re.fullmatch(r'/api/pipeline/datasets/ds-[0-9a-f]{32}/features', endpoint)
            if endpoint not in ('/api/compare', '/api/datasets/load',
                                '/api/pipeline/datasets/generate', '/api/pipeline/datasets/import',
                                '/api/pipeline/train', '/api/pipeline/compare') and not graph_query and not feature_selection:
                self._error(404, 'Unknown API endpoint.')
                return
            if self.headers.get_content_type() != 'application/json':
                self._error(415, 'Send application/json.')
                return
            raw_length = self.headers.get('Content-Length', '')
            if self.headers.get('Transfer-Encoding') or not raw_length.isdecimal():
                self._error(411, 'A Content-Length is required.')
                return
            length = int(raw_length)
            if length < 1 or length > MAX_BODY_BYTES:
                self._error(413, f'Request must contain 1–{MAX_BODY_BYTES} bytes.')
                return
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Incomplete JSON request.')
                def reject_constant(value):
                    raise ValueError('Nonfinite JSON value: ' + value)
                payload = json.loads(body, parse_constant=reject_constant)
                if endpoint.startswith('/api/pipeline/'):
                    if pipeline_service is None:
                        self._error(503, 'The preparation pipeline is unavailable. Start the app with python serve.py.')
                        return
                    status, result = pipeline_service.post(endpoint, payload)
                    self._send(status, json.dumps(result, allow_nan=False).encode('utf-8'))
                    return
                if endpoint == '/api/datasets/load':
                    self._send(200, json.dumps(dataset_service.load(payload), allow_nan=False).encode('utf-8'))
                    return
                validate_request(payload)
                superseded = service.register_request(self.headers.get('X-Comparison-Session'),
                                                      self.headers.get('X-Comparison-Revision'))
                def cancelled():
                    if superseded():
                        return True
                    try:
                        readable, _, _ = select.select([self.connection], [], [], 0)
                        return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b''
                    except OSError:
                        return True
                self._send(200, service.compare_bytes(payload, cancelled))
            except ModelUnavailable as error:
                self._error(503, str(error))
            except InterruptedError as error:
                self._error(409, str(error))
            except FileNotFoundError as error:
                self._error(404, str(error))
            except (ValueError, KeyError, TypeError, UnicodeError, RecursionError) as error:
                self._error(400, str(error))
            except (TimeoutError, ConnectionError):
                self._error(408, 'Request body was not received in time.')
            except Exception:
                traceback.print_exc()
                self._error(500, 'Server operation failed. Check the server output; a new request can be retried.')

    if ':' in host:
        class IPv6Server(ComparisonHTTPServer):
            address_family = socket.AF_INET6
        return IPv6Server((host, port), Handler)
    return ComparisonHTTPServer((host, port), Handler)
