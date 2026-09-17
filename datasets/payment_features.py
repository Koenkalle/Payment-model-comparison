"""Payment feature formulas and compatibility with historical demo checkpoints.

Dataset recipes live in datasets.features. The legacy state adapter below keeps
its original ordered, settled-history behavior for already-trained checkpoints;
new datasets explicitly use timestamp-grouped observed attempts instead.
"""
import math
import numpy as np

AMOUNT_BINS = [15, 35, 75, 150, 300, 650, 1500, 4000, 10000]
AMOUNT = AMOUNT_BINS
# Definitions are ordered to preserve compatibility with exported checkpoints.
FEATURE_SPECS = [
    {
        "id": "log_amount",
        "label": "Payment amount",
        "unit": "currency units",
        "description": "Current requested payment amount.",
        "window": "current payment",
        "transform": "log1p(max(0, value)) / 8",
        "scale": 8,
        "group": "payment",
        "requirements": ["amount"],
    },
    {
        "id": "amount_bin",
        "label": "Amount bucket",
        "unit": "bucket",
        "description": "Zero-based bucket: number of fixed amount-bin edges less than or equal to the "
        "requested amount.",
        "window": "current payment",
        "transform": "bucket index / number of amount-bin edges",
        "scale": None,
        "group": "payment",
        "requirements": ["amount"],
    },
    {
        "id": "sender_out_count",
        "label": "Sender outgoing payments",
        "unit": "payments",
        "description": "Number of strictly earlier observed outgoing payments.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_in_count",
        "label": "Sender incoming payments",
        "unit": "payments",
        "description": "Number of strictly earlier observed incoming payments.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_out_count",
        "label": "Recipient outgoing payments",
        "unit": "payments",
        "description": "Number of strictly earlier observed outgoing payments.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_in_count",
        "label": "Recipient incoming payments",
        "unit": "payments",
        "description": "Number of strictly earlier observed incoming payments.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_out_value",
        "label": "Sender outgoing amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed outgoing payment amounts.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_in_value",
        "label": "Sender incoming amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed incoming payment amounts.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_out_value",
        "label": "Recipient outgoing amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed outgoing payment amounts.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_in_value",
        "label": "Recipient incoming amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed incoming payment amounts.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_seen",
        "label": "Sender observed activities",
        "unit": "activities",
        "description": "Number of strictly earlier observed payments and deposits. Unsettled payment attempts "
        "count as activity; outcome reports do not.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_seen",
        "label": "Recipient observed activities",
        "unit": "activities",
        "description": "Number of strictly earlier observed payments and deposits. Unsettled payment attempts "
        "count as activity; outcome reports do not.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_gap",
        "label": "Sender activity gap",
        "unit": "minutes",
        "description": "Nonnegative minutes since the last observed payment or deposit; outcome reports do "
        "not reset the gap. Zero for first activity.",
        "window": "last observed activity",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_gap",
        "label": "Recipient activity gap",
        "unit": "minutes",
        "description": "Nonnegative minutes since the last observed payment or deposit; outcome reports do "
        "not reset the gap. Zero for first activity.",
        "window": "last observed activity",
        "transform": "log1p(max(0, value)) / 6",
        "scale": 6,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "pair_out_count",
        "label": "Prior payments to recipient",
        "unit": "payments",
        "description": "Strictly earlier observed payments from this sender to this recipient.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 3",
        "scale": 3,
        "group": "account pair",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "pair_total_count",
        "label": "Prior payments in either direction",
        "unit": "payments",
        "description": "Strictly earlier observed payment count summed across both directions of this account "
        "pair.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 3",
        "scale": 3,
        "group": "account pair",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "prior_contact",
        "label": "Prior payment contact",
        "unit": "boolean",
        "description": "One if this pair has a strictly earlier observed payment in either direction; zero "
        "otherwise.",
        "window": "all preceding history",
        "transform": "identity",
        "scale": None,
        "group": "account pair",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_recent_in_count",
        "label": "Sender recent incoming payments",
        "unit": "payments",
        "description": "Strictly earlier observed incoming payments during the preceding 60 minutes, "
        "including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 4",
        "scale": 4,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_recent_out_count",
        "label": "Sender recent outgoing payments",
        "unit": "payments",
        "description": "Strictly earlier observed outgoing payments during the preceding 60 minutes, "
        "including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 4",
        "scale": 4,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_recent_in_count",
        "label": "Recipient recent incoming payments",
        "unit": "payments",
        "description": "Strictly earlier observed incoming payments during the preceding 60 minutes, "
        "including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 4",
        "scale": 4,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_recent_out_count",
        "label": "Recipient recent outgoing payments",
        "unit": "payments",
        "description": "Strictly earlier observed outgoing payments during the preceding 60 minutes, "
        "including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 4",
        "scale": 4,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_recent_in_value",
        "label": "Sender recent incoming amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed incoming payment amounts during the preceding 60 "
        "minutes, including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "sender_recent_out_value",
        "label": "Sender recent outgoing amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed outgoing payment amounts during the preceding 60 "
        "minutes, including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_recent_in_value",
        "label": "Recipient recent incoming amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed incoming payment amounts during the preceding 60 "
        "minutes, including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "recipient_recent_out_value",
        "label": "Recipient recent outgoing amount",
        "unit": "currency units",
        "description": "Sum of strictly earlier observed outgoing payment amounts during the preceding 60 "
        "minutes, including the window boundary.",
        "window": "preceding 60 minutes",
        "transform": "log1p(max(0, value)) / 10",
        "scale": 10,
        "group": "recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "amount_vs_sender_out_mean",
        "label": "Amount / sender outgoing mean",
        "unit": "ratio",
        "description": "Requested amount divided by the sender historical observed outgoing mean. Count and "
        "mean denominators are floored at one; without history this equals the amount.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 8",
        "scale": 8,
        "group": "payment and sender",
        "requirements": ["amount", "identities", "timestamps"],
    },
    {
        "id": "amount_vs_recipient_in_mean",
        "label": "Amount / recipient incoming mean",
        "unit": "ratio",
        "description": "Requested amount divided by the recipient historical observed incoming mean. Count "
        "and mean denominators are floored at one; without history this equals the amount.",
        "window": "all preceding history",
        "transform": "log1p(max(0, value)) / 8",
        "scale": 8,
        "group": "payment and recipient",
        "requirements": ["amount", "identities", "timestamps"],
    },
]
FEATURE_NAMES = [feature["id"] for feature in FEATURE_SPECS]


def _log1p(x, scale=1.0):
    return math.log1p(max(0.0, float(x))) / scale


def legacy_state(n):
    return {
        "last": np.zeros(n),
        "seen": np.zeros(n),
        "oc": np.zeros(n),
        "ic": np.zeros(n),
        "ov": np.zeros(n),
        "iv": np.zeros(n),
        "pairs": np.zeros((n, n)),
        "incidents": [[] for _ in range(n)],
        "now": 0.0,
    }


def legacy_feature_row(state, e):
    u, v, t, amount = int(e["u"]), int(e["v"]), float(e["t"]), float(e["amount"])
    assert u >= 0 and v >= 0

    def recent(n, role, window=60.0):
        events = [
            x for x in state["incidents"][n] if t - x[1] <= window and x[3] == role
        ]
        return len(events), sum(x[2] for x in events)

    si, so = recent(u, 1), recent(u, -1)
    ri, ro = recent(v, 1), recent(v, -1)
    sender_mean = state["ov"][u] / max(1.0, state["oc"][u])
    recipient_mean = state["iv"][v] / max(1.0, state["ic"][v])
    gap_u = max(0.0, t - state["last"][u]) if state["seen"][u] else 0.0
    gap_v = max(0.0, t - state["last"][v]) if state["seen"][v] else 0.0
    return np.array(
        [
            _log1p(amount, 8),
            np.searchsorted(AMOUNT, amount, side="right") / len(AMOUNT),
            _log1p(state["oc"][u], 6),
            _log1p(state["ic"][u], 6),
            _log1p(state["oc"][v], 6),
            _log1p(state["ic"][v], 6),
            _log1p(state["ov"][u], 10),
            _log1p(state["iv"][u], 10),
            _log1p(state["ov"][v], 10),
            _log1p(state["iv"][v], 10),
            _log1p(state["seen"][u], 6),
            _log1p(state["seen"][v], 6),
            _log1p(gap_u, 6),
            _log1p(gap_v, 6),
            _log1p(state["pairs"][u, v], 3),
            _log1p(state["pairs"][u, v] + state["pairs"][v, u], 3),
            float(state["pairs"][u, v] + state["pairs"][v, u] > 0),
            _log1p(si[0], 4),
            _log1p(so[0], 4),
            _log1p(ri[0], 4),
            _log1p(ro[0], 4),
            _log1p(si[1], 10),
            _log1p(so[1], 10),
            _log1p(ri[1], 10),
            _log1p(ro[1], 10),
            _log1p(amount / max(1.0, sender_mean), 8),
            _log1p(amount / max(1.0, recipient_mean), 8),
        ],
        dtype=float,
    )


def legacy_apply(state, e):
    t = float(e["t"])
    u, v = int(e["u"]), int(e["v"])
    settled = e.get("settled", True) is not False
    for n in [v, u] if u >= 0 else [v]:
        state["last"][n] = t
        state["seen"][n] += 1
    if e["kind"] == "payment" and u >= 0 and settled:
        amount = float(e["amount"])
        state["incidents"][u].append((v, t, amount, -1))
        state["incidents"][v].append((u, t, amount, 1))
        state["oc"][u] += 1
        state["ic"][v] += 1
        state["ov"][u] += amount
        state["iv"][v] += amount
        state["pairs"][u, v] += 1
    state["now"] = t
