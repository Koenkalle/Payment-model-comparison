"""Validate the common payment-event document used by the offline browser tools."""
import copy, json
from framework.contracts import EventDataset
from ._validation import source, provenance, number, label


def normalize(raw, metadata=None):
    if not isinstance(raw, dict):
        raise ValueError("Expected a payment-event JSON object.")
    if not isinstance(raw.get("truth", {}), dict):
        raise ValueError("truth must map payment IDs to outcomes.")
    if raw.get("schema", "payment-events/v1") != "payment-events/v1":
        raise ValueError("Unsupported payment dataset schema.")
    units = raw.get("units", {"time": "minutes", "currency": "EUR"})
    if units.get("time") != "minutes" or units.get("currency") != "EUR":
        raise ValueError(
            "Browser payment datasets require minutes and EUR; convert explicitly in a dataset loader."
        )
    accounts = raw.get("accounts")
    events = raw.get("events")
    if (
        not isinstance(accounts, list)
        or not accounts
        or not isinstance(events, list)
        or not events
    ):
        raise ValueError("Payment data needs nonempty accounts and events.")
    normalized_accounts = []
    external = set()
    for index, account in enumerate(accounts):
        if account.get("id") != index:
            raise ValueError(
                "Account IDs must be contiguous indices; use the CSV loader to map external IDs."
            )
        external_id = str(account.get("external_id", account["id"]))
        if external_id in external:
            raise ValueError("Duplicate external account identity.")
        external.add(external_id)
        normalized_accounts.append(
            {
                **account,
                "external_id": external_id,
                "name": str(account.get("name", external_id)),
            }
        )
    normalized = []
    ids = set()
    payment_ids = set()
    last = -float("inf")
    for raw_event in events:
        event = {
            key: raw_event[key]
            for key in ("id", "kind", "t", "u", "v", "amount")
            if key in raw_event
        }
        if set(event) != {"id", "kind", "t", "u", "v", "amount"}:
            raise ValueError("Every event requires id, kind, t, u, v and amount.")
        if not isinstance(event["id"], str) or not event["id"] or event["id"] in ids:
            raise ValueError("Event IDs must be nonempty unique strings.")
        ids.add(event["id"])
        event["t"] = number(event["t"], "event time")
        event["amount"] = number(event["amount"], "amount")
        if event["t"] < 0 or event["t"] < last or event["amount"] < 0:
            raise ValueError(
                "Events must be chronological, with nonnegative times and amounts."
            )
        last = event["t"]
        u, v = event["u"], event["v"]
        if (
            type(u) != int
            or type(v) != int
            or not 0 <= v < len(accounts)
            or not -1 <= u < len(accounts)
            or u == v
        ):
            raise ValueError("Invalid event account index.")
        if event["kind"] not in ("payment", "deposit", "report") or (
            event["kind"] == "payment"
        ) != (u >= 0):
            raise ValueError("Invalid event kind/sender combination.")
        if event["kind"] == "payment":
            payment_ids.add(event["id"])
        if "settled" in raw_event:
            if type(raw_event["settled"]) != bool:
                raise ValueError("settled must be boolean.")
            event["settled"] = raw_event["settled"]
        if "reference" in raw_event:
            event["reference"] = str(raw_event["reference"])
        normalized.append(event)
    truth = {}
    for identifier, outcome in raw.get("truth", {}).items():
        if identifier not in payment_ids:
            raise ValueError("Outcome refers to an unknown payment: " + identifier)
        value = label(outcome)
        if value >= 0:
            truth[identifier] = bool(value)
    availability = raw.get("label_available_at", {})
    if not isinstance(availability, dict):
        raise ValueError(
            "label_available_at must map payment IDs to confirmation times in minutes."
        )
    payment_times = {
        event["id"]: event["t"] for event in normalized if event["kind"] == "payment"
    }
    confirmed = {}
    for identifier, value in availability.items():
        if identifier not in truth:
            raise ValueError(
                "Label availability requires a known payment outcome: " + identifier
            )
        value = number(value, "label availability time")
        if value < payment_times[identifier]:
            raise ValueError(
                "A payment outcome cannot be confirmed before the payment."
            )
        confirmed[identifier] = value
    return {
        "schema": "payment-events/v1",
        "units": dict(units),
        "name": str(raw.get("name", "Imported payments")),
        "size": str(raw.get("size", "imported")),
        "seed": raw.get("seed"),
        "accounts": normalized_accounts,
        "events": normalized,
        "truth": truth,
        "focus": [],
        "bookmarks": [],
        "startIndex": 0,
        "description": str(raw.get("description", "User-supplied payment events.")),
        "provenance": copy.deepcopy(
            metadata or raw.get("provenance", {"origin": "user-supplied"})
        ),
        **({"label_available_at": confirmed} if availability else {}),
    }


def load(config, base):
    path = source(config, base)
    metadata = provenance(config, path)
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("provenance"), dict):
        metadata["source_provenance"] = copy.deepcopy(raw["provenance"])
    document = normalize(raw, metadata)
    return EventDataset(document, metadata)
