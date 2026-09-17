"""Bounded, model-independent inspection of immutable dataset snapshots.

Providers translate source identities and relationships into a small graph
contract. A disposable SQLite index stores topology and inspection properties;
neither graph feature extraction nor a full in-memory graph is needed. Extend
``PROVIDERS`` for a new storage format. New canonical stream adapters already
work through their descriptor's source and destination roles.

Outcome labels are evaluation metadata, exposed explicitly for inspection.
They are never copied into observable features or sent to training code here.
"""
from contextlib import closing
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import threading
import uuid

from datasets.stream import open_prepared
from datasets.views import _selection


SCHEMA = 'dataset-graph/v1'
INDEX_VERSION = 1
MAX_NODES = 500
MAX_EDGES = 2000
MAX_SEARCH = 50


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def node_id(kind, external_id):
    """Preserve type and identity without delimiter collisions or numeric casts."""
    return _json([str(kind), str(external_id)])


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    type: str
    properties: dict


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source: str
    target: str
    type: str
    time: float
    label: int
    properties: dict


class PreparedStreamProvider:
    """Use canonical entities/features; label diagnostics stay in the truth store."""

    def populate(self, config, writer):
        with open_prepared(config['path']) as stream:
            descriptor = stream.descriptor
            graph = descriptor.get('graph')
            if not graph:
                raise ValueError('This dataset has no entity relationships for a graph view.')
            for batch in stream.iter_batches(512, **_selection(config)):
                outcomes = stream.truth_for(event.event_id for event in batch)
                for event in batch:
                    references = {entity.role: entity for entity in event.entities}
                    source = references[graph['source_role']]
                    target = references[graph['destination_role']]
                    for entity in event.entities:
                        writer.node(GraphNode(node_id(entity.kind, entity.id), entity.id,
                                              entity.kind, {'external_id': entity.id}))
                    outcome = outcomes[event.event_id]
                    properties = {'features': dict(event.features), 'currency': descriptor.get('currency'),
                                  'outcome': {'label': outcome.label, 'available_at': outcome.available_at}}
                    writer.edge(GraphEdge(event.event_id, node_id(source.kind, source.id),
                                          node_id(target.kind, target.id), event.event_type,
                                          event.event_time, outcome.label, properties))


class PaymentDocumentProvider:
    """Project payment relationships and declared accounts, including isolates.

    Deposits and fraud reports have no account source and are not fabricated
    into relationships. The stored payment document is already size bounded
    and validated by DatasetStore; reading it does not run a model or replay.
    """

    def populate(self, config, writer):
        document = json.loads(Path(config['path']).read_text(encoding='utf-8'))
        identities = {}
        for account in document['accounts']:
            external_id = str(account.get('external_id', account['id']))
            kind = str(account.get('entity_type', 'account'))
            identifier = node_id(kind, external_id)
            identities[account['id']] = identifier
            # Whitelist observable identity fields; account annotations may
            # contain scenario truth or internal source metadata.
            properties = {'external_id': external_id}
            writer.node(GraphNode(identifier, str(account.get('name', external_id)), kind, properties))
        truth = document.get('truth', {})
        availability = document.get('label_available_at', {})
        for event in document['events']:
            if event['kind'] != 'payment':
                continue
            label = int(truth[event['id']]) if event['id'] in truth else -1
            confirmed = availability.get(event['id'])
            properties = {'features': {'amount': event['amount']}, 'currency': 'EUR',
                          'outcome': {'label': label, 'available_at': None if confirmed is None else confirmed * 60}}
            writer.edge(GraphEdge(event['id'], identities[event['u']], identities[event['v']],
                                  'payment', event['t'] * 60, label, properties))


PROVIDERS = {'prepared_fraud': PreparedStreamProvider, 'payment_json': PaymentDocumentProvider}


class _IndexWriter:
    def __init__(self, database):
        self.database = database
        self.sequence = 0

    def node(self, node):
        self.database.execute('INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?,?,0,0)',
                              (node.id, node.label, node.type, _json(node.properties),
                               node.label.casefold(), str(node.properties.get('external_id', node.label)).casefold()))

    def edge(self, edge):
        self.sequence += 1
        self.database.execute('INSERT INTO edges VALUES (?,?,?,?,?,?,?,?)',
                              (self.sequence, edge.id, edge.source, edge.target, edge.type,
                               edge.time, edge.label, _json(edge.properties)))
        self.database.execute('UPDATE nodes SET out_degree=out_degree+1 WHERE id=?', (edge.source,))
        self.database.execute('UPDATE nodes SET in_degree=in_degree+1 WHERE id=?', (edge.target,))
        if edge.source == edge.target:
            self.database.execute('INSERT INTO adjacency VALUES (?,?,1,1)', (edge.source, self.sequence))
        else:
            self.database.execute('INSERT INTO adjacency VALUES (?,?,0,1)', (edge.source, self.sequence))
            self.database.execute('INSERT INTO adjacency VALUES (?,?,1,0)', (edge.target, self.sequence))


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer from {minimum} to {maximum}.')
    return value


def _string(value, name, maximum=4096):
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f'{name} must be a nonempty string of at most {maximum} characters.')
    return value


def _query(payload):
    allowed = {'mode', 'node_id', 'offset', 'node_limit', 'edge_limit', 'direction',
               'start', 'stop', 'label', 'edge_type', 'existing_node_ids'}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError('Unknown graph query fields.')
    query = {**payload}
    query['mode'] = payload.get('mode', 'sample')
    if query['mode'] not in ('sample', 'neighbors'):
        raise ValueError('mode must be sample or neighbors.')
    query['direction'] = payload.get('direction', 'both')
    if query['direction'] not in ('both', 'incoming', 'outgoing'):
        raise ValueError('direction must be both, incoming or outgoing.')
    if query['mode'] == 'neighbors':
        _string(query.get('node_id'), 'node_id')
    elif 'node_id' in query:
        raise ValueError('node_id is only supported for a neighbors query.')
    for field, default, minimum, maximum in [('offset', 0, 0, 2**63 - 1),
                                            ('node_limit', 100, 1, MAX_NODES),
                                            ('edge_limit', 300, 1, MAX_EDGES)]:
        query[field] = _integer(payload.get(field, default), field, minimum, maximum)
    for field in ('start', 'stop'):
        if field in query:
            try:
                finite = type(query[field]) in (float, int) and math.isfinite(query[field])
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(field + ' must be finite seconds.')
    if 'start' in query and 'stop' in query and query['start'] > query['stop']:
        raise ValueError('start must not exceed stop.')
    if 'label' in query:
        _integer(query['label'], 'label', -1, 1)
    if 'edge_type' in query:
        _string(query['edge_type'], 'edge_type', 200)
    existing = payload.get('existing_node_ids', [])
    if not isinstance(existing, list) or len(existing) > query['node_limit']:
        raise ValueError('existing_node_ids must be an array within the node limit.')
    for identifier in existing:
        _string(identifier, 'existing_node_ids entry')
    if len(set(existing)) != len(existing):
        raise ValueError('existing_node_ids must be unique.')
    query['existing_node_ids'] = existing
    return query


class DatasetGraphService:
    """Read/query derived topology with a rebuildable, versioned disk cache.

    Snapshot checksums are verified once per file stat signature. Repeated
    interactions read only the graph index, rather than hashing or parsing the
    complete source again. Atomic replacement allows concurrent readers to
    keep their own short-lived read-only SQLite connections.
    """

    def __init__(self, store, providers=None):
        self.store = store
        self.providers = dict(PROVIDERS if providers is None else providers)
        self.directory = store.root / 'graph-index'
        self.directory.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._validated = {}

    @staticmethod
    def _signature(path):
        if path.is_symlink():
            raise ValueError('Invalid stored dataset payload.')
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _config(self, identifier):
        directory = self.store._directory(identifier)
        # Both possible payload names are fixed by DatasetStore, never supplied
        # by the HTTP caller. Include metadata so cache keys track snapshot edits.
        signature = tuple((name, self._signature(directory / name)) for name in
                          ('metadata.json', 'config.json', 'transactions.sqlite', 'payments.json')
                          if (directory / name).exists())
        previous = self._validated.get(identifier)
        if previous is None or previous[0] != signature:
            config = self.store._config(identifier)
            self._validated[identifier] = (signature, config)
        return self._validated[identifier][1]

    def _index(self, identifier):
        with self._lock:
            metadata = self.store.get(identifier)
            if not {'graph', 'labeled_graph'}.intersection(metadata.get('views', [])):
                raise ValueError('This dataset has no entity relationships for a graph view.')
            config = self._config(identifier)
            provider = self.providers.get(config['loader'])
            if provider is None:
                raise ValueError('This storage format does not yet provide graph inspection.')
            key = _json([INDEX_VERSION, metadata.get('payload_sha256'), metadata.get('config_sha256'),
                         metadata.get('fingerprint')])
            path = self.directory / (identifier + '.sqlite')
            if path.is_symlink():
                raise ValueError('Invalid graph index.')
            if path.exists():
                try:
                    with closing(self._connect(path)) as database:
                        row = database.execute('SELECT value FROM metadata WHERE key="cache_key"').fetchone()
                        if row and row[0] == key:
                            return path
                except sqlite3.Error:
                    pass  # A corrupt disposable index can be rebuilt from the snapshot.
            self._build(path, provider(), config, key)
            return path

    @staticmethod
    def _connect(path):
        database = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        database.row_factory = sqlite3.Row
        return database

    def _build(self, path, provider, config, key):
        temporary = path.with_name('.building-' + uuid.uuid4().hex + '.sqlite')
        try:
            with closing(sqlite3.connect(temporary)) as database:
                database.executescript('''
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE nodes (id TEXT PRIMARY KEY, label TEXT NOT NULL, type TEXT NOT NULL,
                        properties TEXT NOT NULL, search_label TEXT NOT NULL, search_external TEXT NOT NULL,
                        in_degree INTEGER NOT NULL, out_degree INTEGER NOT NULL);
                    CREATE TABLE edges (seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                        source TEXT NOT NULL, target TEXT NOT NULL, type TEXT NOT NULL, time REAL NOT NULL,
                        label INTEGER NOT NULL, properties TEXT NOT NULL);
                    CREATE TABLE adjacency (node_id TEXT NOT NULL, seq INTEGER NOT NULL,
                        incoming INTEGER NOT NULL, outgoing INTEGER NOT NULL, PRIMARY KEY(node_id,seq));
                ''')
                with database:
                    provider.populate(config, _IndexWriter(database))
                    database.executescript('''
                        CREATE INDEX nodes_label ON nodes(search_label,id);
                        CREATE INDEX nodes_external ON nodes(search_external,id);
                        CREATE INDEX edges_label ON edges(label,seq);
                        CREATE INDEX edges_type ON edges(type,seq);
                        CREATE INDEX edges_time ON edges(time,seq);
                        CREATE INDEX adjacency_incoming ON adjacency(node_id,seq) WHERE incoming=1;
                        CREATE INDEX adjacency_outgoing ON adjacency(node_id,seq) WHERE outgoing=1;
                    ''')
                    database.row_factory = sqlite3.Row
                    counts = {'nodes': database.execute('SELECT count(*) FROM nodes').fetchone()[0],
                              'edges': database.execute('SELECT count(*) FROM edges').fetchone()[0]}
                    interval = database.execute('SELECT min(time),max(time) FROM edges').fetchone()
                    summary = {'supported': True, 'schema': SCHEMA, 'counts': counts,
                               'node_types': [dict(row) for row in database.execute(
                                   'SELECT type,count(*) AS count FROM nodes GROUP BY type ORDER BY type')],
                               'edge_types': [dict(row) for row in database.execute(
                                   'SELECT type,count(*) AS count FROM edges GROUP BY type ORDER BY type')],
                               'time_range': {'start': interval[0], 'end': interval[1], 'unit': 'seconds'},
                               'limits': {'max_nodes': MAX_NODES, 'max_edges': MAX_EDGES},
                               'label_semantics': 'Evaluation outcomes for inspection; not model predictions.',
                               'degree_semantics': 'All relationships in the saved dataset, before view filters.'}
                    database.executemany('INSERT INTO metadata VALUES (?,?)',
                                         [('cache_key', key), ('summary', _json(summary))])
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def summary(self, identifier):
        metadata = self.store.get(identifier)
        if not {'graph', 'labeled_graph'}.intersection(metadata.get('views', [])):
            return {'supported': False, 'schema': SCHEMA,
                    'reason': 'This dataset has no entity identities or relationships. Its numeric rows remain available.'}
        config = self._config(identifier)
        if config['loader'] not in self.providers:
            return {'supported': False, 'schema': SCHEMA,
                    'reason': 'This storage format does not yet provide graph inspection.'}
        with closing(self._connect(self._index(identifier))) as database:
            return json.loads(database.execute('SELECT value FROM metadata WHERE key="summary"').fetchone()[0])

    @staticmethod
    def _node(row):
        return {'id': row['id'], 'label': row['label'], 'type': row['type'],
                'properties': json.loads(row['properties']), 'in_degree': row['in_degree'],
                'out_degree': row['out_degree'], 'degree': row['in_degree'] + row['out_degree']}

    def _nodes(self, database, identifiers):
        if not identifiers:
            return []
        rows = database.execute('SELECT * FROM nodes WHERE id IN (' + ','.join('?' for _ in identifiers) + ')',
                                list(identifiers)).fetchall()
        found = {row['id']: self._node(row) for row in rows}
        if len(found) != len(identifiers):
            raise ValueError('Unknown graph node ID.')
        return [found[identifier] for identifier in identifiers]

    def query(self, identifier, payload):
        query = _query(payload)
        with closing(self._connect(self._index(identifier))) as database:
            selected = dict.fromkeys(query['existing_node_ids'])
            if query['mode'] == 'neighbors':
                selected[query['node_id']] = None
            if len(selected) > query['node_limit']:
                raise ValueError('The selected node and existing nodes exceed node_limit.')
            self._nodes(database, selected)  # Reject fabricated identities before scanning relationships.
            clauses, parameters = ['e.seq > ?'], [query['offset']]
            for field, condition in [('start', 'e.time >= ?'), ('stop', 'e.time < ?'),
                                     ('label', 'e.label = ?'), ('edge_type', 'e.type = ?')]:
                if field in query:
                    clauses.append(condition)
                    parameters.append(query[field])
            source = 'edges e'
            if query['mode'] == 'neighbors':
                source = 'adjacency a JOIN edges e ON e.seq=a.seq'
                clauses.append('a.node_id = ?')
                parameters.append(query['node_id'])
                if query['direction'] != 'both':
                    clauses.append('a.' + query['direction'] + '=1')
            sql = 'SELECT e.* FROM ' + source + ' WHERE ' + ' AND '.join(clauses) + ' ORDER BY e.seq LIMIT ?'
            rows = database.execute(sql, [*parameters, query['edge_limit'] + 1]).fetchall()
            edges, cursor, reason = [], query['offset'], None
            for row in rows:
                new_nodes = {row['source'], row['target']} - selected.keys()
                if len(selected) + len(new_nodes) > query['node_limit']:
                    reason = 'node_limit'
                    break
                if len(edges) == query['edge_limit']:
                    reason = 'edge_limit'
                    break
                selected.setdefault(row['source'], None)
                selected.setdefault(row['target'], None)
                cursor = row['seq']
                edges.append({key: row[key] for key in ('id', 'source', 'target', 'type', 'time', 'label')})
                edges[-1]['properties'] = json.loads(row['properties'])
            # An unfiltered empty graph/sample can still expose declared isolates.
            if not selected and query['mode'] == 'sample' and not rows and query['offset'] == 0 and not any(
                    field in query for field in ('start', 'stop', 'label', 'edge_type')):
                selected.update((row['id'], None) for row in database.execute(
                    'SELECT id FROM nodes ORDER BY id LIMIT ?', (query['node_limit'],)))
            nodes = self._nodes(database, selected)
            has_more = reason is not None
            return {'nodes': nodes, 'edges': edges, 'page': {'offset': query['offset'],
                    'next_offset': cursor if has_more and cursor > query['offset'] else None, 'has_more': has_more},
                    'counts': {'nodes': len(nodes), 'edges': len(edges)}, 'truncated': has_more,
                    'limit_reason': reason}

    def search(self, identifier, query='', limit=20):
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError('q must be a string of at most 200 characters.')
        _integer(limit, 'limit', 1, MAX_SEARCH)
        with closing(self._connect(self._index(identifier))) as database:
            prefix = query.strip().casefold()
            found = {}
            for column in ('search_label', 'search_external'):
                rows = database.execute('SELECT * FROM nodes WHERE ' + column + ' >= ? AND ' + column +
                                        ' < ? ORDER BY ' + column + ',id LIMIT ?',
                                        (prefix, prefix + '\U0010ffff', limit + 1))
                for row in rows:
                    found[row['id']] = self._node(row)
            exact = database.execute('SELECT * FROM nodes WHERE id=?', (query,)).fetchone()
            if exact:
                found[exact['id']] = self._node(exact)
            nodes = sorted(found.values(), key=lambda node: (node['label'].casefold(), node['id']))
            return {'nodes': nodes[:limit], 'has_more': len(nodes) > limit}
