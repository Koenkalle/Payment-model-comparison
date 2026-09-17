"""Fixed random-projection temporal prototypes; only the optional logistic head is trained."""
from ._common import *
from .xgboost_numpy import xgb_state, xgb_feature_row, xgb_apply
from ._neural import supervised_probability


def sequence_config(model):
    return model.get("sequence", {})


def sequence_history_vector(model, state, n, t):
    cfg = sequence_config(model)
    dim = int(cfg.get("dim", 8))
    weights = np.asarray(cfg.get("history_weights", np.zeros((dim, 8))), dtype=float)
    out = np.zeros(dim)
    total = 0.0
    events = list(state["incidents"][n])[-int(cfg.get("history_limit", 8)) :][::-1]
    for other, when, amount, role in events:
        age = max(0.0, float(t) - float(when))
        weight = math.exp(-age / float(cfg.get("history_decay", 720)))
        basis = np.asarray(
            [
                1.0,
                float(role),
                math.log1p(amount) / 8.0,
                min(age / 720.0, 4.0),
                math.sin(age / 60.0),
                math.cos(age / 60.0),
                math.sin(age / 1440.0),
                math.cos(age / 1440.0),
            ]
        )
        out += weight * (weights @ basis)
        total += weight
    return np.tanh(out / total) if total else out


def sequence_pair_events(model, state, u, v):
    cfg = sequence_config(model)
    limit = int(cfg.get("pair_limit", 6))
    return [e for e in state["incidents"][u] if e[0] == v and e[3] == -1][-limit:][::-1]


def sequence_pair_vector(model, state, u, v, t):
    cfg = sequence_config(model)
    dim = int(cfg.get("pair_dim", 4))
    weights = np.asarray(cfg.get("pair_weights", np.zeros((dim, 5))), dtype=float)
    out = np.zeros(dim)
    total = 0.0
    for other, when, amount, role in sequence_pair_events(model, state, u, v):
        age = max(0.0, float(t) - float(when))
        weight = math.exp(-age / float(cfg.get("pair_decay", 1440)))
        basis = np.asarray(
            [
                1.0,
                math.log1p(amount) / 8.0,
                min(age / 720.0, 4.0),
                math.sin(age / 60.0),
                math.cos(age / 60.0),
            ]
        )
        out += weight * (weights @ basis)
        total += weight
    return np.tanh(out / total) if total else out


def sequence_neighbors(model, state, n):
    limit = int(sequence_config(model).get("neighbor_limit", 4))
    seen = set()
    out = []
    for event in list(state["incidents"][n])[-limit:][::-1]:
        if event[0] not in seen:
            seen.add(event[0])
            out.append(event)
    return out


def sequence_graph_vector(model, state, n, t, depth, cache):
    cfg = sequence_config(model)
    key = (n, depth)
    if key in cache:
        return cache[key]
    base = sequence_history_vector(model, state, n, t)
    if not cfg.get("use_gnn") or depth <= 0:
        cache[key] = base
        return base
    dim = int(cfg.get("dim", 8))
    weights = np.asarray(cfg.get("gnn_weights", np.zeros((dim, dim + 4))), dtype=float)
    out = base.copy()
    total = 0.0
    for other, when, amount, role in sequence_neighbors(model, state, n):
        child = sequence_graph_vector(model, state, other, t, depth - 1, cache)
        age = max(0.0, float(t) - float(when))
        edge = np.asarray(
            [float(role), math.log1p(amount) / 8.0, min(age / 720.0, 4.0), 1.0]
        )
        value = np.tanh(weights @ np.concatenate([child, edge]))
        weight = math.exp(-age / float(cfg.get("history_decay", 720)))
        out += weight * value
        total += weight
    cache[key] = np.tanh(base + (out - base) / total) if total else base
    return cache[key]


def sequence_account_embedding(model, state, n, t, cache):
    cfg = sequence_config(model)
    return (
        sequence_graph_vector(model, state, n, t, int(cfg.get("gnn_layers", 2)), cache)
        if cfg.get("use_gnn")
        else sequence_history_vector(model, state, n, t)
    )


def sequence_amount_logits(model, state, u, v, t):
    cfg = sequence_config(model)
    edges = model["amount_bins"]
    hist = np.zeros(len(edges) + 1)
    decay = float(cfg.get("history_decay", 720))
    for other, when, amount, role in state["incidents"][u]:
        if role == -1:
            hist[int(np.searchsorted(edges, amount, side="right"))] += math.exp(
                -max(0.0, t - when) / decay
            )
    if cfg.get("use_pair"):
        for other, when, amount, role in sequence_pair_events(model, state, u, v):
            hist[int(np.searchsorted(edges, amount, side="right"))] += float(
                cfg.get("pair_amount_weight", 1.0)
            ) * math.exp(-max(0.0, t - when) / float(cfg.get("pair_decay", 1440)))
    return np.log(hist + float(cfg.get("histogram_smoothing", 0.5)))


def sequence_gap_logits(model, state, u, v, t):
    cfg = sequence_config(model)
    edges = model["gap_bins"]
    hist = np.zeros(len(edges) + 2)
    out = sorted([e for e in state["incidents"][u] if e[3] == -1], key=lambda e: e[1])
    last = None
    for other, when, amount, role in out:
        if last is not None:
            hist[
                int(np.searchsorted(edges, max(0.0, when - last), side="right"))
            ] += math.exp(-max(0.0, t - when) / float(cfg.get("history_decay", 720)))
        last = when
    if cfg.get("use_pair"):
        pair = sorted(sequence_pair_events(model, state, u, v), key=lambda e: e[1])
        last = None
        for other, when, amount, role in pair:
            if last is not None:
                hist[
                    int(np.searchsorted(edges, max(0.0, when - last), side="right"))
                ] += float(cfg.get("pair_gap_weight", 1.0)) * math.exp(
                    -max(0.0, t - when) / float(cfg.get("pair_decay", 1440))
                )
            last = when
    return np.log(hist + float(cfg.get("histogram_smoothing", 0.5)))


def sequence_softmax(values):
    values = np.asarray(values, dtype=float)
    shift = values - values.max()
    exp = np.exp(shift)
    return exp / exp.sum()


def sequence_prediction(model, state, e):
    cfg = sequence_config(model)
    n = len(state["last"])
    u = int(e["u"])
    v = int(e["v"])
    t = float(e["t"])
    cache = {}
    sender = sequence_account_embedding(model, state, u, t, cache)
    history_sender = sequence_history_vector(model, state, u, t)
    dim = max(1, len(sender))
    recipient = np.full(n, -1e9)
    for candidate in range(n):
        if candidate == u:
            continue
        candidate_embedding = sequence_account_embedding(
            model, state, candidate, t, cache
        )
        history_candidate = sequence_history_vector(model, state, candidate, t)
        pair = (
            sequence_pair_vector(model, state, u, candidate, t)
            if cfg.get("use_pair")
            else np.zeros(int(cfg.get("pair_dim", 4)))
        )
        temporal = float(np.dot(history_sender, history_candidate) / dim)
        graph = (
            float(np.dot(sender, candidate_embedding) / dim)
            if cfg.get("use_gnn")
            else 0.0
        )
        recipient[candidate] = (
            float(cfg.get("recipient_pair_weight", 0))
            * math.log1p(state["pairs"][u, candidate])
            + float(cfg.get("recipient_activity_weight", 0))
            * math.log1p(state["ic"][candidate])
            + float(cfg.get("recipient_temporal_weight", 0)) * temporal
            + float(cfg.get("recipient_pair_state_weight", 0))
            * (pair[0] if len(pair) else 0)
            + float(cfg.get("recipient_graph_weight", 0)) * graph
        )
    amount = sequence_amount_logits(model, state, u, v, t)
    gap_logits = sequence_gap_logits(model, state, u, v, t)
    probabilities = [
        sequence_softmax(recipient),
        sequence_softmax(amount),
        sequence_softmax(gap_logits),
    ]
    gap = max(0.0, t - state["last"][u]) if state["seen"][u] else None
    buckets = [
        v,
        int(np.searchsorted(model["amount_bins"], e["amount"], side="right")),
        int(np.searchsorted(model["gap_bins"], gap, side="right"))
        if gap is not None
        else len(model["gap_bins"]) + 1,
    ]
    parts = [
        -math.log2(max(1e-30, float(probabilities[i][buckets[i]]))) for i in range(3)
    ]
    return {
        "score": float(sum(parts)),
        "parts": parts,
        "probabilities": probabilities,
        "buckets": buckets,
        "gap": gap,
        "embedding": [sender, sequence_account_embedding(model, state, v, t, cache)],
        "features": xgb_feature_row(state, e).tolist(),
    }


def sequence_supervised_rows(model, dataset):
    rows = []
    labels = []
    episodes = []
    for episode_index, episode in enumerate(dataset["episodes"]):
        state = xgb_state(episode["accounts"])
        for e in episode["events"]:
            if e["kind"] == "payment" and int(e.get("label", -1)) >= 0:
                prediction = sequence_prediction(model, state, e)
                rows.append(
                    np.asarray(
                        prediction["features"] + [prediction["score"]], dtype=float
                    )
                )
                labels.append(int(e["label"]))
                episodes.append(episode_index)
            xgb_apply(state, e)
    return (
        np.asarray(rows, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(episodes, dtype=int),
    )


def fit_sequence_supervised_head(model, dataset, steps=220, learning_rate=0.08):
    X, y, episode_ids = sequence_supervised_rows(model, dataset)
    split = max(1, int(len(dataset["episodes"]) * 0.8))
    train = episode_ids < split
    valid = ~train
    if train.sum() == 0 or y[train].sum() == 0:
        raise ValueError("Supervised mode requires at least one flagged fraud example.")
    mean = X[train].mean(0)
    scale = X[train].std(0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    Xs = (X[train] - mean) / scale
    yt = y[train]
    positive = max(1.0, float(yt.sum()))
    negative = max(1.0, float(len(yt) - yt.sum()))
    pos_weight = min(10.0, negative / positive)
    weights = np.where(yt > 0, pos_weight, 1.0)
    rate = float(np.clip(yt.mean(), 1e-4, 1 - 1e-4))
    w = np.zeros(Xs.shape[1])
    b = math.log(rate / (1 - rate))
    for _ in range(int(steps)):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w + b, -50, 50)))
        error = (p - yt) * weights
        w -= learning_rate * (Xs.T @ error / max(1, len(yt)))
        b -= learning_rate * float(error.mean())
    raw = (X[valid] - mean) / scale @ w + b
    vp = 1 / (1 + np.exp(-np.clip(raw, -50, 50)))
    vy = y[valid]
    val_loss = float(
        -(
            vy * np.log(np.maximum(vp, 1e-12))
            + (1 - vy) * np.log(np.maximum(1 - vp, 1e-12))
        ).mean()
    )
    return {
        "type": "logistic_fraud_head",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
        "training": {
            "fraud_labels_used": int(yt.sum()),
            "fraud_flags_requested": int(dataset.get("flags_requested", 0)),
            "validation_flags_used": int(dataset.get("validation_flags_used", 0)),
            "training_rows": int(train.sum()),
            "validation_rows": int(valid.sum()),
            "steps": int(steps),
            "learning_rate": learning_rate,
            "scale_pos_weight": pos_weight,
            "heldout_logloss": val_loss,
            "notice": "Supervised head trained offline from flagged historical outcomes; scores are not calibrated production fraud probabilities.",
        },
    }


def sequence_trace(model, episode, head=None):
    state = xgb_state(episode["accounts"])
    records = []
    empty = [[[] for _ in range(episode["accounts"])]]
    for e in episode["events"]:
        before = [[] for _ in range(episode["accounts"])]
        if e["kind"] == "payment":
            prediction = sequence_prediction(model, state, e)
            base = prediction["score"]
            if head is not None:
                row = prediction["features"] + [base]
                p = supervised_probability(head, row)
                score = -math.log2(max(1e-12, 1 - p))
                probabilities = [[[1 - p, p]], [[1.0]], [[1.0]]]
            else:
                score = base
                probabilities = [
                    [prediction["probabilities"][i].tolist()] for i in range(3)
                ]
            records.append(
                {
                    "scores": [score],
                    "probabilities": probabilities,
                    "memory_before": [before],
                    "embedding": [
                        prediction["embedding"][0].tolist(),
                        prediction["embedding"][1].tolist(),
                    ],
                }
            )
        else:
            records.append(
                {
                    "scores": [0.0],
                    "probabilities": [[[1.0]], [[1.0]], [[1.0]]],
                    "memory_before": [before],
                    "embedding": [[], []],
                }
            )
        xgb_apply(state, e)
        records[-1]["memory_after"] = empty
    return records


def sequence_payload(a, design):
    from framework.registry import model_entry, load_module

    variant = DESIGNS[design]["variant"]
    cfg = dict(load_module(model_entry(design)["python_module"]).CONFIG)
    cfg.update(
        {
            "dim": 8,
            "pair_dim": 4,
            "history_limit": 8,
            "pair_limit": 6,
            "neighbor_limit": 4,
            "gnn_layers": 2,
            "history_decay": 720.0,
            "pair_decay": 1440.0,
            "pair_amount_weight": 1.5,
            "pair_gap_weight": 1.5,
            "histogram_smoothing": 0.5,
        }
    )
    seed = a.model_seed + sum((i + 1) * ord(c) for i, c in enumerate(design))
    rng = np.random.default_rng(seed)
    cfg["history_weights"] = rng.normal(0, 0.24, (8, 8)).tolist()
    cfg["pair_weights"] = rng.normal(0, 0.24, (4, 5)).tolist()
    cfg["gnn_weights"] = rng.normal(0, 0.18, (8, 12)).tolist()
    return {
        "version": 3,
        "id": design,
        "adapter": "temporal_family",
        "family": "temporal_family",
        "label": DESIGNS[design]["label"],
        "architecture": DESIGNS[design],
        "hidden": 0,
        "neighbors": int(cfg["neighbor_limit"]),
        "amount_bins": AMOUNT,
        "gap_bins": GAP,
        "sequence": cfg,
        "training": {
            "method": "Causal temporal-history likelihood with bounded sinusoidal time encoding",
            "variant": variant,
            "fraud_labels_used": 0,
            "default_warmup_requests": int(a.warmup),
            "model_seed": int(seed),
            "history_limit": int(cfg["history_limit"]),
            "pair_limit": int(cfg["pair_limit"]),
            "gnn_layers": int(cfg["gnn_layers"] if cfg.get("use_gnn") else 0),
            "notice": "No fraud labels enter the temporal-history score; optional flagged-history mode adds a separately trained offline fraud head.",
        },
        "policy": {"default_warmup_requests": int(a.warmup)},
    }


def train_sequence_design(a, data, design, labeled=None):
    payload = sequence_payload(a, design)
    supervised = (
        fit_sequence_supervised_head(payload, labeled) if labeled is not None else None
    )
    payload["supervised"] = supervised
    cfg = payload["sequence"]
    payload["parameters"] = sum(
        np.asarray(cfg[k]).size
        for k in ["history_weights", "pair_weights", "gnn_weights"]
    )
    payload["checkpoint_id"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:16]
    (ROOT / "models" / f"{design}.json").write_text(
        json.dumps(payload, separators=(",", ":"))
    )
    episode = data[-1]
    (ROOT / "parity" / f"{design}.json").write_text(
        json.dumps(
            {"episode": episode, "expected": sequence_trace(payload, episode)},
            separators=(",", ":"),
        )
    )
    if supervised is not None:
        (ROOT / "parity-supervised").mkdir(exist_ok=True)
        (ROOT / "parity-supervised" / f"{design}.json").write_text(
            json.dumps(
                {
                    "episode": episode,
                    "expected": sequence_trace(payload, episode, supervised),
                },
                separators=(",", ":"),
            )
        )
    print(
        design,
        "temporal-history checkpoint",
        "parameters",
        payload["parameters"],
        "flags",
        supervised["training"]["fraud_labels_used"] if supervised else 0,
        flush=True,
    )
