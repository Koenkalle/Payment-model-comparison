"""Chronological groups and reproducible destination corruption for link prediction."""
import numpy as np


def timestamp_groups(dataset, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return
    if np.any(np.diff(indices) <= 0):
        raise ValueError('Graph indices must be unique and chronological.')
    cuts = np.flatnonzero(np.diff(dataset.times[indices]) != 0) + 1
    yield from np.split(indices, cuts)


class DestinationSampler:
    """Candidates observed by this timestamp, excluding self and same-time positives.

    Historical pairs remain eligible: a previously observed link need not recur.
    No future links are consulted and sampled negatives never update graph state.
    """
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.candidates = set()

    def observe(self, dataset, group):
        self.candidates.update(map(int, dataset.destinations[group]))

    def sample(self, dataset, group):
        self.observe(dataset, group)
        forbidden = {}
        for i in group:
            forbidden.setdefault(int(dataset.sources[i]), set()).add(int(dataset.destinations[i]))
        candidates = sorted(self.candidates)
        destinations = []
        for i in group:
            u = int(dataset.sources[i])
            valid = [v for v in candidates if v != u and v not in forbidden[u]]
            # -1 means this positive has no valid counterexample yet. It still
            # enters history; exclude it from paired link loss/metrics.
            destinations.append(int(self.rng.choice(valid)) if valid else -1)
        return np.asarray(destinations, dtype=np.int64)
