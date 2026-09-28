import pytest
from multicomm_ensemble.textroute import validate, enrich, SchemaViolationError


def _base():
    return dict(sender_role="coder", receiver_role="reviewer", intent="inform",
                priority=3, body="done", expects_reply=True)


def test_valid_message_passes():
    msg = validate(_base())
    assert msg.schema_version == "1.3"


def test_missing_intent_blocked():
    payload = _base(); payload.pop("intent")
    with pytest.raises(SchemaViolationError) as e:
        validate(payload)
    assert "intent" in str(e.value)


def test_priority_bounds():
    payload = _base(); payload["priority"] = 9
    with pytest.raises(SchemaViolationError):
        validate(payload)


def test_enrich_adds_hash_and_seq():
    d = enrich(validate(_base()), seq=7)
    assert d["seq"] == 7 and len(d["body_sha256"]) == 16
