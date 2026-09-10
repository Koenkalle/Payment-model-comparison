"""Strictly prior first-hop histories for the upstream DyGFormer interface."""
import numpy as np


class NeighborSampler:
    sample_neighbor_strategy = 'recent'

    def __init__(self, dataset, indices=None, history_limit=None):
        self.history_limit = history_limit
        rows = [[] for _ in dataset.node_ids]
        indices = range(len(dataset.ids)) if indices is None else indices
        for i in indices:
            u, v = int(dataset.sources[i]), int(dataset.destinations[i])
            item = (float(dataset.times[i]), i + 1)
            rows[u].append((*item, v))
            rows[v].append((*item, u))
        self.histories = []
        for history in rows:
            history.sort(key=lambda item: (item[0], item[1]))
            self.histories.append((np.asarray([h[2] for h in history], dtype=np.int64),
                                   np.asarray([h[1] for h in history], dtype=np.int64),
                                   np.asarray([h[0] for h in history], dtype=np.float64)))

    def get_all_first_hop_neighbors(self, node_ids, node_interact_times):
        neighbors, edges, times = [], [], []
        for node, timestamp in zip(node_ids, node_interact_times):
            n, e, t = self.histories[int(node)]
            stop = np.searchsorted(t, timestamp, side='left')
            start = max(0, stop - self.history_limit) if self.history_limit else 0
            neighbors.append(n[start:stop].copy())
            edges.append(e[start:stop].copy())
            times.append(t[start:stop].copy())
        return neighbors, edges, times
