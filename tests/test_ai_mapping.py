"""Tests for src/onboarding/ai_mapping.py - the AI-assisted second pass
over suggest_mapping()'s output. Mocked at the _get_client() boundary
(same style as tests/test_copilot_agent.py mocking agent.chat/agent.tools
rather than the real Gemini SDK internals) so these never make a real
network call."""

from __future__ import annotations

import json

from src.onboarding import ai_mapping
from src.onboarding.schema_fields import suggest_mapping


class _FakeResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)


class _FakeModels:
    def __init__(self, payload=None, raise_error=None):
        self._payload = payload
        self._raise_error = raise_error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_error is not None:
            raise self._raise_error
        return _FakeResponse(self._payload)


class _FakeClient:
    def __init__(self, payload=None, raise_error=None):
        self.models = _FakeModels(payload=payload, raise_error=raise_error)


def _stamp_source_only(suggestions):
    """What refine_mapping_with_ai() must return when it never calls the
    AI at all (no key, nothing unresolved, or a hard failure) - every
    entry gets `source` added, nothing else changes."""
    return {
        column: {**info, "source": "name_match" if info.get("matched_known_field") else "none"}
        for column, info in suggestions.items()
    }


# --- No API key: must silently fall back, never raise ---


def test_no_api_key_falls_back_to_name_match_only(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ai_mapping.reset_client_for_testing()

    columns = ["customerID", "some_weird_unmatched_column"]
    base = suggest_mapping(columns)
    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    assert result == _stamp_source_only(base)
    assert result["customerID"]["source"] == "name_match"
    assert result["some_weird_unmatched_column"]["source"] == "none"
    assert result["some_weird_unmatched_column"]["suggested_role"] == "ignore"


# --- Nothing unresolved: must never call the model at all (cost guard) ---


def test_skips_api_call_when_everything_already_matched(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    ai_mapping.reset_client_for_testing()
    fake_client = _FakeClient(payload=[])
    monkeypatch.setattr(ai_mapping, "_get_client", lambda: fake_client)

    columns = ["customerID", "Churn", "MonthlyCharges"]
    base = suggest_mapping(columns)
    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    assert fake_client.models.calls == []
    assert result == _stamp_source_only(base)


# --- A real, well-formed AI response merges in correctly ---


def test_merges_ai_suggestion_for_unresolved_column(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    ai_mapping.reset_client_for_testing()

    columns = ["Client Reference Number", "Widget Zorp Flag"]
    base = suggest_mapping(columns)  # both land on "ignore" - no alias match
    assert base["Client Reference Number"]["suggested_role"] == "ignore"
    assert base["Widget Zorp Flag"]["suggested_role"] == "ignore"

    fake_client = _FakeClient(
        payload=[
            {"column": "Client Reference Number", "suggested_role": "customer_id", "reasoning": "unique per-row id"},
            {"column": "Widget Zorp Flag", "suggested_role": "feature", "reasoning": "plausible predictive signal"},
        ]
    )
    monkeypatch.setattr(ai_mapping, "_get_client", lambda: fake_client)

    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    assert result["Client Reference Number"]["suggested_role"] == "customer_id"
    assert result["Client Reference Number"]["source"] == "ai"
    assert result["Client Reference Number"]["ai_reasoning"] == "unique per-row id"
    assert result["Widget Zorp Flag"]["suggested_role"] == "feature"
    assert result["Widget Zorp Flag"]["source"] == "ai"
    assert len(fake_client.models.calls) == 1


# --- Singleton-role safety: AI must never be allowed to duplicate a role
# suggest_mapping() already confidently claimed, even if it tries to ---


def test_does_not_let_ai_reassign_an_already_claimed_singleton_role(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    ai_mapping.reset_client_for_testing()

    columns = ["customerID", "some_unmatched_column"]
    base = suggest_mapping(columns)
    assert base["customerID"]["suggested_role"] == "customer_id"

    # A misbehaving/hallucinating model tries to claim customer_id again
    # for a second column - must be rejected, not silently accepted.
    fake_client = _FakeClient(
        payload=[{"column": "some_unmatched_column", "suggested_role": "customer_id", "reasoning": "looks like an id"}]
    )
    monkeypatch.setattr(ai_mapping, "_get_client", lambda: fake_client)

    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    # customer_id is not offered to the model at all once claimed (see
    # offerable_roles filtering) - "customer_id" shouldn't even appear in
    # the response_schema enum sent for this call.
    sent_config = fake_client.models.calls[0]["config"]
    offered_roles = sent_config.response_schema["items"]["properties"]["suggested_role"]["enum"]
    assert "customer_id" not in offered_roles
    # And even though the fake model ignored that and returned it anyway,
    # refine_mapping_with_ai() must still refuse to merge a role that
    # wasn't in the offered set.
    assert result["some_unmatched_column"]["suggested_role"] == "ignore"
    assert result["some_unmatched_column"]["source"] == "none"


# --- Malformed/unexpected responses must degrade silently, never raise ---


def test_non_json_response_falls_back_silently(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    ai_mapping.reset_client_for_testing()

    class _BadResponse:
        text = "not valid json {{{"

    class _BadModels:
        def generate_content(self, **kwargs):
            return _BadResponse()

    class _BadClient:
        models = _BadModels()

    monkeypatch.setattr(ai_mapping, "_get_client", lambda: _BadClient())

    columns = ["some_unmatched_column"]
    base = suggest_mapping(columns)
    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    assert result == _stamp_source_only(base)


def test_non_list_response_falls_back_silently(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    ai_mapping.reset_client_for_testing()
    fake_client = _FakeClient(payload={"not": "a list"})
    monkeypatch.setattr(ai_mapping, "_get_client", lambda: fake_client)

    columns = ["some_unmatched_column"]
    base = suggest_mapping(columns)
    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    assert result == _stamp_source_only(base)


def test_api_error_falls_back_silently(monkeypatch):
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    ai_mapping.reset_client_for_testing()

    fake_client = _FakeClient(raise_error=errors.ServerError(500, {"error": {"message": "boom"}}))
    monkeypatch.setattr(ai_mapping, "_get_client", lambda: fake_client)

    columns = ["some_unmatched_column"]
    base = suggest_mapping(columns)
    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    assert result == _stamp_source_only(base)


# --- An AI suggestion of a role outside the offered set is rejected, not
# blindly trusted (defense in depth beyond the response_schema enum) ---


def test_rejects_ai_role_not_in_offered_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    ai_mapping.reset_client_for_testing()

    fake_client = _FakeClient(
        payload=[{"column": "some_unmatched_column", "suggested_role": "duration_leakage_column"}]
    )
    monkeypatch.setattr(ai_mapping, "_get_client", lambda: fake_client)

    columns = ["some_unmatched_column"]
    base = suggest_mapping(columns)
    result = ai_mapping.refine_mapping_with_ai(columns, [{}], base)

    assert result["some_unmatched_column"]["suggested_role"] == "ignore"
    assert result["some_unmatched_column"]["source"] == "none"
