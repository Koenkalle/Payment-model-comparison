"""Sparse graph recipes over strictly preceding observed payment attempts.

The caller scores a complete timestamp group before committing it with
``observe_batch``. Queries never introduce candidate nodes or edges. Repeated
payments count as one directed edge; deposits, reports, incomplete endpoints,
and self-payments do not enter this graph.

PageRank uses amortized sparse snapshots. Personalized PageRank (PPR) instead
uses bounded local pushes on undirected contacts, which also connect entities
in bipartite customer/terminal datasets. PPR values are lower bounds, with a
separate absolute error bound that includes work-budget truncation.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import heapq
import math
from types import MappingProxyType

import numpy as np


DAMPING = .85
PAGERANK_MAX_ITERATIONS = 30
PAGERANK_TOLERANCE = 1e-8
PAGERANK_REFRESH_MIN_EDGES = 1
PAGERANK_REFRESH_FRACTION = .1
PPR_EDGE_BUDGET = 512
PPR_TOLERANCE = 1e-4
PPR_CACHE_SIZE = 128

_HISTORY = 'strictly earlier observed payments; unique edges; self-payments ignored'
_PAGERANK = {
    'method': 'Sparse PageRank power iteration with warm-started, amortized snapshots',
    'orientation': 'directed unique historical payments',
    'parameters': {'damping': DAMPING, 'maximum_iterations': PAGERANK_MAX_ITERATIONS,
                   'l1_tolerance': PAGERANK_TOLERANCE, 'teleport': 'uniform over historical nodes',
                   'dangling_nodes': 'uniform redistribution',
                   'refresh_min_new_edges': PAGERANK_REFRESH_MIN_EDGES,
                   'refresh_new_edge_fraction': PAGERANK_REFRESH_FRACTION},
    'approximation': 'The numerical L1 bound applies to the stored snapshot only. '
                     'New directed edges since that snapshot are reported separately as edge lag. '
                     'Nodes first observed after the snapshot have score zero until refresh.',
}
_PPR = {
    'method': 'Deterministic local forward-push personalized PageRank',
    'orientation': 'undirected unique historical contacts',
    'parameters': {'damping': DAMPING, 'restart_probability': 1 - DAMPING,
                   'edge_traversal_budget': PPR_EDGE_BUDGET,
                   'target_remaining_mass': PPR_TOLERANCE, 'source_cache_size': PPR_CACHE_SIZE},
    'approximation': 'A nonnegative lower bound on current-graph PPR. Its accompanying absolute '
                     'error bound includes all unexpanded continuation mass. The work budget can '
                     'stop calculation before the target tolerance is reached. Unknown endpoints '
                     'have value and bound zero. Not a payment-direction score.',
}


def _spec(identifier, label, description, *, unit='probability', transform='identity',
          readout='', method=None):
    return {'id': identifier, 'label': label, 'description': description,
            'group': 'graph', 'unit': unit, 'transform': transform,
            'window': _HISTORY, 'requirements': ['identities', 'timestamps'],
            'kind': 'derived', 'version': '1', 'readout': readout,
            **(method or {'method': 'Exact incremental contact graph',
                          'orientation': 'undirected unique historical contacts',
                          'parameters': {}, 'approximation': 'Exact for all preceding history.'})}


GRAPH_FEATURE_SPECS = [
    _spec('graph_sender_pagerank', 'Sender PageRank',
          'Global importance of the sender in the latest historical directed payment snapshot.',
          readout='Sender stationary probability; zero if absent from the snapshot.', method=_PAGERANK),
    _spec('graph_recipient_pagerank', 'Recipient PageRank',
          'Global importance of the recipient in the latest historical directed payment snapshot.',
          readout='Recipient stationary probability; zero if absent from the snapshot.', method=_PAGERANK),
    _spec('graph_ppr_sender_to_recipient', 'Sender-to-recipient contact PPR',
          'Local graph proximity to the recipient for random walks restarting at the sender. '
          'Contact edges can be traversed in either direction, including through shared terminals.',
          readout='Lower bound on recipient probability with restart at the sender.', method=_PPR),
    _spec('graph_ppr_recipient_to_sender', 'Recipient-to-sender contact PPR',
          'Local graph proximity to the sender for random walks restarting at the recipient. '
          'The restart endpoint changes; contacts remain undirected.',
          readout='Lower bound on sender probability with restart at the recipient.', method=_PPR),
    _spec('graph_ppr_forward_error_bound', 'Sender contact PPR error bound',
          'Conservative absolute error bound for sender-to-recipient contact PPR; '
          'the exact probability lies between its score and score plus this bound.',
          readout='Unresolved probability mass after bounded local exploration.', method=_PPR),
    _spec('graph_ppr_reverse_error_bound', 'Recipient contact PPR error bound',
          'Conservative absolute error bound for recipient-to-sender contact PPR; '
          'large values indicate that the work budget limited the approximation.',
          readout='Unresolved probability mass after bounded local exploration.', method=_PPR),
    _spec('graph_pagerank_error_bound', 'PageRank numerical error bound',
          'Global L1 distance bound between the stored PageRank vector and the exact stationary '
          'distribution on that snapshot. It does not bound drift from subsequent edges.',
          readout='Snapshot-wide L1 bound, which also bounds any individual endpoint error.', method=_PAGERANK),
    _spec('graph_pagerank_edge_lag', 'PageRank snapshot edge lag',
          'Number of unique directed payment edges added since the latest PageRank snapshot.',
          unit='directed edges', readout='Zero means PageRank uses the current historical topology.',
          method=_PAGERANK),
    _spec('graph_same_component', 'Same historical contact component',
          'Whether both endpoints already belong to the same weakly connected payment component. '
          'An endpoint absent from earlier payments yields zero.', unit='indicator',
          readout='One for an existing path of contacts, zero otherwise.'),
    _spec('graph_sender_component_size', 'Sender contact component size',
          'Number of historical identities in the sender\'s weakly connected component.',
          unit='identities', transform='log1p(value)', readout='Natural log of one plus component size.'),
    _spec('graph_recipient_component_size', 'Recipient contact component size',
          'Number of historical identities in the recipient\'s weakly connected component.',
          unit='identities', transform='log1p(value)', readout='Natural log of one plus component size.'),
    _spec('graph_sender_degree', 'Sender distinct contacts',
          'Number of distinct historical counterparties of the sender, in either payment direction.',
          unit='contacts', transform='log1p(value)', readout='Natural log of one plus distinct contact degree.'),
    _spec('graph_recipient_degree', 'Recipient distinct contacts',
          'Number of distinct historical counterparties of the recipient, in either payment direction.',
          unit='contacts', transform='log1p(value)', readout='Natural log of one plus distinct contact degree.'),
]


def _identity_key(identity):
    """Canonical ordering without merging typed or compound identities."""
    if isinstance(identity, tuple):
        return ('tuple', tuple(_identity_key(value) for value in identity))
    return (type(identity).__name__, repr(identity))


@dataclass(frozen=True)
class PPRResult:
    """A cached source approximation and auditable work/error accounting.

    ``reserve`` is a sub-probability vector. ``estimate(target)`` also credits
    the known one-hop stopping contribution of a partially expanded hub, even
    when that target falls outside the traversed neighbor prefix. References
    to append-only neighbor ordinals avoid copying a hub's complete adjacency;
    the captured degree excludes all neighbors added after this calculation.
    """
    reserve: object
    remaining_mass: float
    edge_traversals: int
    dropped_mass: float = 0.
    _lookup: object = field(default_factory=dict, repr=False)
    _tail_contacts: object = field(default_factory=dict, repr=False)
    _tail_degree: int = field(default=0, repr=False)
    _tail_expanded: object = field(default_factory=frozenset, repr=False)
    _tail_stopping_share: float = field(default=0., repr=False)

    def _extra(self, target):
        node = self._lookup.get(target)
        ordinal = self._tail_contacts.get(node)
        if ordinal is not None and ordinal < self._tail_degree and node not in self._tail_expanded:
            return self._tail_stopping_share
        return 0.

    def estimate(self, target):
        return self.reserve.get(target, 0.) + self._extra(target)

    def error_bound(self, target=None):
        return max(0., self.remaining_mass - self._extra(target))


class GraphFeatureState:
    """Append-only sparse topology with bounded local graph-query work.

    ``observe_batch`` must receive only events whose whole timestamp group has
    already been scored. ``begin_group`` refreshes a due PageRank snapshot once;
    ``values`` also calls it for direct users and never changes graph topology.
    """

    def __init__(self, *, damping=DAMPING, pagerank_max_iterations=PAGERANK_MAX_ITERATIONS,
                 pagerank_tolerance=PAGERANK_TOLERANCE, ppr_edge_budget=PPR_EDGE_BUDGET,
                 ppr_tolerance=PPR_TOLERANCE, ppr_cache_size=PPR_CACHE_SIZE):
        if not 0 < damping < 1:
            raise ValueError('Graph damping must lie strictly between zero and one.')
        if pagerank_max_iterations < 1 or ppr_edge_budget < 1 or ppr_cache_size < 1:
            raise ValueError('Graph iteration, traversal and cache budgets must be positive.')
        if pagerank_tolerance <= 0 or ppr_tolerance <= 0:
            raise ValueError('Graph approximation tolerances must be positive.')
        self.damping = float(damping)
        self.pagerank_max_iterations = int(pagerank_max_iterations)
        self.pagerank_tolerance = float(pagerank_tolerance)
        self.ppr_edge_budget = int(ppr_edge_budget)
        self.ppr_tolerance = float(ppr_tolerance)
        self.ppr_cache_size = int(ppr_cache_size)
        self.nodes = []
        self.node_lookup = {}
        self.outgoing = []
        self.contacts = []  # dict neighbor index -> append-only insertion ordinal
        self.parent = []
        self.component_sizes = []
        self.edge_count = 0
        self.contact_version = 0
        self.pagerank_refreshes = 0
        self.ppr_computations = 0
        self._pagerank = np.empty(0, dtype=float)
        self._pagerank_error = 0.
        self._pagerank_edge_count = 0
        self._ppr_cache = OrderedDict()

    def _root(self, node):
        # Union by size keeps depth logarithmic; queries avoid path mutations.
        while node != self.parent[node]:
            node = self.parent[node]
        return node

    def _union(self, source, destination):
        left, right = self._root(source), self._root(destination)
        if left == right:
            return
        if (self.component_sizes[left], -left) < (self.component_sizes[right], -right):
            left, right = right, left
        self.parent[right] = left
        self.component_sizes[left] += self.component_sizes[right]

    def observe_batch(self, events):
        edges = {(event.source, event.destination) for event in events
                 if event.kind == 'payment' and event.source is not None
                 and event.destination is not None and event.source != event.destination}
        if not edges:
            return
        identities = {identity for edge in edges for identity in edge}
        unseen = (identity for identity in identities if identity not in self.node_lookup)
        for identity in sorted(unseen, key=_identity_key):
            node = len(self.nodes)
            self.node_lookup[identity] = node
            self.nodes.append(identity)
            self.outgoing.append(set())
            self.contacts.append({})
            self.parent.append(node)
            self.component_sizes.append(1)
        contacts_changed = False
        for source, destination in sorted((self.node_lookup[source], self.node_lookup[destination])
                                          for source, destination in edges):
            if destination in self.outgoing[source]:
                continue
            self.outgoing[source].add(destination)
            self.edge_count += 1
            if destination not in self.contacts[source]:
                self.contacts[source][destination] = len(self.contacts[source])
                self.contacts[destination][source] = len(self.contacts[destination])
                self._union(source, destination)
                contacts_changed = True
        if contacts_changed:
            self.contact_version += 1
            self._ppr_cache.clear()

    def begin_group(self):
        if not self.edge_count:
            return
        refresh_after = max(PAGERANK_REFRESH_MIN_EDGES,
                            math.ceil(PAGERANK_REFRESH_FRACTION * self._pagerank_edge_count))
        if not self._pagerank.size or self.edge_count - self._pagerank_edge_count >= refresh_after:
            self._refresh_pagerank()

    def _refresh_pagerank(self):
        count = len(self.nodes)
        degree = np.fromiter((len(neighbors) for neighbors in self.outgoing), dtype=np.int64, count=count)
        sources = np.repeat(np.arange(count), degree)
        targets = np.fromiter((target for neighbors in self.outgoing for target in sorted(neighbors)),
                              dtype=np.int64, count=self.edge_count)
        if self._pagerank.size:
            rank = np.pad(self._pagerank, (0, count - len(self._pagerank)))
        else:
            rank = np.full(count, 1. / count)
        safe_degree = np.maximum(degree, 1)
        dangling = degree == 0
        error = 2.
        for _ in range(self.pagerank_max_iterations):
            inbound = np.bincount(targets, weights=(rank / safe_degree)[sources], minlength=count)
            updated = self.damping * inbound + ((1 - self.damping) +
                        self.damping * rank[dangling].sum()) / count
            delta = np.abs(updated - rank).sum()
            # Contraction certificate for the returned iterate, not the previous one.
            error = min(2., self.damping / (1 - self.damping) * delta + 1e-14)
            rank = updated
            if error <= self.pagerank_tolerance:
                break
        self._pagerank = rank
        self._pagerank_error = float(error)
        self._pagerank_edge_count = self.edge_count
        self.pagerank_refreshes += 1

    def compute_ppr(self, source):
        """Return a source-cached local lower bound, without admitting nodes."""
        node = self.node_lookup.get(source)
        if node is None:
            return PPRResult(MappingProxyType({}), 0., 0)
        cached = self._ppr_cache.get(node)
        if cached is not None:
            self._ppr_cache.move_to_end(node)
            return cached
        result = self._push_ppr(node)
        self.ppr_computations += 1
        self._ppr_cache[node] = result
        if len(self._ppr_cache) > self.ppr_cache_size:
            self._ppr_cache.popitem(last=False)
        return result

    def _push_ppr(self, source):
        residual, reserve = {source: 1.}, {}
        pending = [(-1., source)]
        unresolved, used, dropped = 1., 0, 0.
        tail_contacts, tail_expanded, tail_degree, tail_share = {}, frozenset(), 0, 0.
        stop = 1 - self.damping
        while pending and unresolved > self.ppr_tolerance and used < self.ppr_edge_budget:
            negative_mass, node = heapq.heappop(pending)
            mass = -negative_mass
            if residual.get(node, 0.) != mass:
                continue  # superseded heap entry
            del residual[node]
            stopping = stop * mass
            reserve[node] = reserve.get(node, 0.) + stopping
            unresolved -= stopping
            neighbors = self.contacts[node]
            degree = len(neighbors)
            share = self.damping * mass / degree
            limit = min(degree, self.ppr_edge_budget - used)
            expanded = []
            for neighbor in neighbors:
                if len(expanded) == limit:
                    break
                residual[neighbor] = residual.get(neighbor, 0.) + share
                heapq.heappush(pending, (-residual[neighbor], neighbor))
                expanded.append(neighbor)
            used += limit
            if limit < degree:
                dropped += share * (degree - limit)
                tail_contacts, tail_degree = neighbors, degree
                tail_expanded, tail_share = frozenset(expanded), stop * share
                break
        # Every pending random walk may stop immediately without traversing an
        # edge. Credit that known mass; all continuation remains in the bound.
        for node, mass in residual.items():
            reserve[node] = reserve.get(node, 0.) + stop * mass
        remaining = min(1., max(0., 1 - math.fsum(reserve.values())) + 1e-14)
        by_identity = {self.nodes[node]: value for node, value in reserve.items()}
        return PPRResult(MappingProxyType(by_identity), remaining, used, dropped,
                         self.node_lookup, tail_contacts, tail_degree, tail_expanded, tail_share)

    def values(self, source, destination):
        self.begin_group()
        left, right = self.node_lookup.get(source), self.node_lookup.get(destination)
        left_root = self._root(left) if left is not None else None
        right_root = self._root(right) if right is not None else None
        connected = left_root is not None and left_root == right_root
        # Different exact components (including unseen endpoints) prove PPR zero
        # and avoid running a local approximation at all for impossible paths.
        if connected:
            forward, reverse = self.compute_ppr(source), self.compute_ppr(destination)
            forward_value, reverse_value = forward.estimate(destination), reverse.estimate(source)
            forward_error, reverse_error = forward.error_bound(destination), reverse.error_bound(source)
        else:
            forward_value = reverse_value = forward_error = reverse_error = 0.
        return {
            'graph_sender_pagerank': float(self._pagerank[left]) if left is not None and left < len(self._pagerank) else 0.,
            'graph_recipient_pagerank': float(self._pagerank[right]) if right is not None and right < len(self._pagerank) else 0.,
            'graph_ppr_sender_to_recipient': forward_value,
            'graph_ppr_recipient_to_sender': reverse_value,
            'graph_ppr_forward_error_bound': forward_error,
            'graph_ppr_reverse_error_bound': reverse_error,
            'graph_pagerank_error_bound': self._pagerank_error,
            'graph_pagerank_edge_lag': float(self.edge_count - self._pagerank_edge_count),
            'graph_same_component': float(connected),
            'graph_sender_component_size': math.log1p(self.component_sizes[left_root]) if left_root is not None else 0.,
            'graph_recipient_component_size': math.log1p(self.component_sizes[right_root]) if right_root is not None else 0.,
            'graph_sender_degree': math.log1p(len(self.contacts[left])) if left is not None else 0.,
            'graph_recipient_degree': math.log1p(len(self.contacts[right])) if right is not None else 0.,
        }
