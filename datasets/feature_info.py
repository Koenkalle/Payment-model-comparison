"""Presentation-only explanations for dataset inputs.

The saved recipe remains authoritative for formulas, units and history. These
notes enrich its metadata at read time, so improving help never recalculates
values or changes a dataset fingerprint. A new recipe may supply its own
``info`` object with description, sections and facts.
"""
import copy
import re

from .payment_features import FEATURE_SPECS


# Older caches can contain presentation placeholders in either description
# location. They are not provider-authored semantics and must not mask a useful
# saved definition when help is enriched at read time.
_PLACEHOLDERS = {
    "numeric model input",
    "numeric model feature",
    "numeric input supplied to the model from this dataset",
    "original numeric dataset column, preserved without modification",
    "an original numeric column preserved from the dataset",
    "a stored numeric model input",
    "a stored input column in this saved dataset",
}


def _placeholder(value):
    return not value or str(value).strip().rstrip(".").lower() in _PLACEHOLDERS


_PAYMENT_DESCRIPTIONS = {
    feature["id"]: feature["description"] for feature in FEATURE_SPECS
}
_PAYMENT_INTRODUCTIONS = {
    "log_amount": "Size of the current requested payment, compressed with the saved logarithmic transformation so large transfers do not dominate the numeric scale.",
    "amount_bin": "Where the requested payment amount falls among the fixed amount boundaries. Payments in the same bucket receive the same value.",
    "sender_seen": "How much earlier activity has been observed for this sender, including payments sent, payments received and deposits. Unsettled attempts count; outcome reports do not.",
    "recipient_seen": "How much earlier activity has been observed for this recipient, including payments sent, payments received and deposits. Unsettled attempts count; outcome reports do not.",
    "sender_gap": "How long this sender has been inactive: minutes since its last observed payment or deposit. A first-time sender has a gap of zero.",
    "recipient_gap": "How long this recipient has been inactive: minutes since its last observed payment or deposit. A first-time recipient has a gap of zero.",
    "pair_out_count": "Earlier payment attempts from the current sender to this particular recipient. Payments in the reverse direction do not contribute.",
    "pair_total_count": "Earlier payment attempts between the current sender and recipient, counting both directions. Repeated payments contribute separately.",
    "prior_contact": "Whether the current sender and recipient have interacted before: one for an earlier payment attempt in either direction, zero when no such contact exists.",
    "amount_vs_sender_out_mean": "How large this payment is relative to the amounts this sender previously sent. It compares the requested amount with the sender’s historical outgoing average.",
    "amount_vs_recipient_in_mean": "How large this payment is relative to the amounts this recipient previously received. It compares the requested amount with the recipient’s historical incoming average.",
}
for _endpoint in ("sender", "recipient"):
    for _direction, _action in (("out", "sent"), ("in", "received")):
        _PAYMENT_INTRODUCTIONS[f"{_endpoint}_{_direction}_count"] = (
            f"How many payment attempts this {_endpoint} {_action} at strictly earlier timestamps. "
            "Repeated payments to or from the same counterparty count separately."
        )
        _PAYMENT_INTRODUCTIONS[f"{_endpoint}_{_direction}_value"] = (
            f"Total requested amount of payment attempts this {_endpoint} {_action} at strictly earlier "
            "timestamps. This measures accumulated payment value rather than the number of payments."
        )
        _PAYMENT_INTRODUCTIONS[f"{_endpoint}_recent_{_direction}_count"] = (
            f"How many payment attempts this {_endpoint} {_action} in the preceding 60 minutes. "
            "It highlights recent activity, includes the window boundary and excludes the current timestamp."
        )
        _PAYMENT_INTRODUCTIONS[f"{_endpoint}_recent_{_direction}_value"] = (
            f"Total requested amount of payment attempts this {_endpoint} {_action} in the preceding "
            "60 minutes. It measures recent payment value, including the window boundary but excluding the current timestamp."
        )


_SOURCE_NOTES = {
    "amount": (
        "Requested payment amount",
        "The payment amount supplied by the source, before any model preprocessing.",
        "Larger values represent larger payments in the stated currency. The feature itself "
        "does not convert currency or express how unusual an amount is for this account.",
    ),
    "log1p_amount": (
        "Log payment amount",
        "The natural logarithm of one plus the requested amount.",
        "This compresses large amounts while preserving their order. Zero amount gives "
        "zero; a value of 1 represents an amount of about 1.72 source currency units. "
        "This column is not divided by the scale used by the separate log_amount recipe.",
    ),
    "prior_source_count": (
        "Prior sender payments",
        "The number of outgoing payments observed for this sender at strictly earlier timestamps.",
        "This is an untransformed count. Zero means no earlier outgoing payment in "
        "the stored history. Repeated payments count separately; it is not a count of distinct contacts.",
    ),
    "prior_destination_count": (
        "Prior recipient payments",
        "The number of incoming payments observed for this recipient at strictly earlier timestamps.",
        "This is an untransformed count. Zero means no earlier incoming payment in "
        "the stored history. Payments at the current timestamp are excluded.",
    ),
    "oldbalanceOrg": (
        "Sender balance before payment",
        "The origin account balance recorded before the transaction in the PaySim source.",
        "This is the supplied pre-transaction balance, not a balance reconstructed by this tool. "
        "Read it in the source currency and alongside the payment amount.",
    ),
    "oldbalanceDest": (
        "Recipient balance before payment",
        "The destination account balance recorded before the transaction in the PaySim source.",
        "This preserves the supplied value. A zero can reflect how the source represents "
        "an account, so do not interpret it alone as evidence of suspicious behavior.",
    ),
}


def source_definition(name, currency=None):
    """Describe an existing source column without reading or calculating rows."""
    label, description, _ = _source_note(name)
    definition = {
        "id": name,
        "label": label,
        "description": description,
        "kind": "source",
        "group": "source",
        "unit": "source units",
        "transform": "identity",
        "window": "source row",
        "requirements": [],
        "version": "1",
        "available": True,
    }
    if name in ("amount", "log1p_amount", "oldbalanceOrg", "oldbalanceDest"):
        definition["unit"] = currency or "source currency units (unspecified)"
    if name == "log1p_amount":
        definition.update(transform="log1p(amount)", window="current payment")
    if name in ("prior_source_count", "prior_destination_count"):
        definition.update(unit="payments", window="strictly earlier payment timestamps")
    return describe_feature(definition)


def _source_note(name):
    if name in _SOURCE_NOTES:
        return _SOURCE_NOTES[name]
    if re.fullmatch(r"V(?:[1-9]|1[0-9]|2[0-8])", name):
        return (
            name + " · anonymized input",
            f"ULB’s {name} input, preserved as supplied. Its underlying transaction attribute is anonymized, so no business meaning can be assigned to this particular column.",
            "Its original business meaning is not available to this tool. Positive and negative "
            "values describe the supplied numeric representation, not a fraud probability or payment amount.",
        )
    return (
        name.replace("_", " "),
        f"Source column “{name}”, preserved exactly as supplied. Its meaning and units must be read from the source schema.",
        "The tool has no further semantic definition for this source column. Consult the source "
        "schema before interpreting its magnitude. Its values remain unchanged by feature selection.",
    )


def _interpretation(feature):
    name = feature["id"]
    if feature.get("kind") == "source":
        return _source_note(name)[2]
    if name in ("log_amount", "log1p_amount"):
        return (
            "A larger value means a larger requested amount. The logarithm compresses very large "
            "payments; this is an amount input, not a model probability. Read the calculation for its scale."
        )
    if name == "amount_bin":
        return (
            "Crossing a fixed bin edge increases the bucket. Amounts within a bin have the same "
            "value. The edges are not fitted to this dataset, so their usefulness depends on the currency and amount range."
        )
    if name.startswith("graph_"):
        if "error_bound" in name:
            return (
                "A smaller bound means greater numerical certainty. A large bound calls for caution "
                "when interpreting the corresponding estimate; it does not indicate fraud. "
                "PageRank snapshot drift is separate from its numerical error."
            )
        if "edge_lag" in name:
            return (
                "Zero means the PageRank snapshot includes all historical directed relationships. "
                "A positive value counts newly observed relationships waiting for the next refresh, not elapsed time."
            )
        if "pagerank" in name:
            return (
                "PageRank measures global structural importance: incoming relationships from important "
                "accounts contribute more. Scores are probability mass and depend on graph size. "
                "An unseen endpoint has no historical score; high importance is not a fraud probability."
            )
        if "ppr_" in name:
            return (
                "Personalized PageRank measures proximity through contact paths from the restart endpoint. "
                "More probability means stronger proximity under this random-walk model. Compare the score "
                "with its error bound: a small lower bound with a large error bound is unresolved, not proof of no relationship."
            )
        if "same_component" in name:
            return (
                "One means an earlier contact path already connects the endpoints, even if they never "
                "paid each other directly. Zero means separate components or absent history. Direction is ignored."
            )
        if "component_size" in name:
            return (
                "Larger values mean the endpoint belongs to a larger connected contact network. The "
                "stored value is log1p of the entity count. A large component can include many normal customers and merchants."
            )
        if "degree" in name:
            return (
                "This counts distinct counterparties in either payment direction and stores log1p of "
                "that count. Repeated payments to the same counterparty do not increase the degree."
            )
    if name == "prior_contact":
        return (
            "One means this pair has an earlier observed payment in either direction; zero means "
            "no such payment in the available history. It indicates familiarity, not whether a payment is safe."
        )
    if name.startswith("pair_"):
        return (
            "Larger values mean more earlier payments between this pair in the stated direction or "
            "directions. Repeated payments count separately. The transformation compresses the raw count."
        )
    if "amount_vs_" in name:
        return (
            "Before the logarithmic transformation, a ratio above one means the amount exceeds the "
            "historical mean, subject to the denominator floor. With no prior history the raw ratio "
            "equals the amount, so it is not a reliable relative-size comparison for a new account."
        )
    if name.endswith("_gap"):
        return (
            "A larger value means a longer pause since earlier activity. Zero can mean no prior "
            "activity or a zero gap, so use it with an activity count to distinguish those cases."
        )
    if name.endswith("_seen"):
        return (
            "This measures how much observed account activity is available, including payments and "
            "deposits. It is not the number of counterparties and does not count outcome reports."
        )
    if name.endswith(("_count", "_value")):
        measure = "total amount" if name.endswith("_value") else "number of payments"
        window = (
            "the previous 60 minutes, which can highlight a burst of activity"
            if "_recent_" in name
            else "all available preceding history, which reflects accumulated activity"
        )
        return (
            f"A larger value means a larger {measure} over {window}. Incoming and outgoing refer "
            "to the named endpoint. The displayed calculation transforms the raw total; zero means no "
            "contributing history or a zero total."
        )
    return (
        "Read this value using the supplied calculation, units and history window. Its magnitude is "
        "an input to a model, not a prediction by itself. Compare held-out runs to assess whether it helps."
    )


def describe_feature(definition):
    """Enrich a copy; never rewrite stored formulas, statistics or provenance."""
    result = copy.deepcopy(definition)
    description = result.get("description") or result.get("help")
    if result.get("kind") == "source" and _placeholder(description):
        description = _source_note(result["id"])[1]
    elif (
        result.get("kind") != "source"
        and result["id"] in _PAYMENT_INTRODUCTIONS
        and description == _PAYMENT_DESCRIPTIONS[result["id"]]
    ):
        # Enrich recognized bundled wording only. A saved custom description or
        # changed historical recipe continues to describe its own values.
        description = _PAYMENT_INTRODUCTIONS[result["id"]]
    elif _placeholder(description):
        description = (
            f'Stored feature “{result["id"]}”. Read its saved calculation and history '
            "below; no further semantic explanation was supplied for this definition."
        )
    notes = {
        "description": description,
        "sections": [{"title": "How to read it", "text": _interpretation(result)}],
    }
    if result.get("kind") != "source":
        notes["sections"].append(
            {
                "title": "Using this input",
                "text": "Enable or disable this column independently when saving a feature dataset. Its stored "
                "values are shared across models. A feature can be useful in combination with other inputs "
                "even when it has little effect on its own.",
            }
        )
    supplied = copy.deepcopy(result.get("info", {}))
    if (
        _placeholder(supplied.get("description"))
        or supplied.get("description") == result.get("description")
        and description != result.get("description")
    ):
        supplied.pop("description", None)
    notes.update(supplied)
    result["info"] = notes
    return result
