"""Stage 2 tests: the Gemini wiring itself (schemas, dispatch, tenant_id
injection, unavailable-key handling) - NOT the adversarial honesty suite,
that's tests/test_copilot_guardrails.py (Stage 3) and requires a real
GEMINI_API_KEY to exercise actual model responses."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base
from src.copilot import agent


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _get_token(client, tenant_id="telco"):
    email = f"copilot-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


# --- Missing/invalid API key must never produce a fabricated response ---


def test_chat_raises_copilot_unavailable_when_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent.reset_client_for_testing()
    with pytest.raises(agent.CopilotUnavailableError):
        agent.chat("hello", tenant_id="telco")


def test_chat_endpoint_returns_503_when_key_missing(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent.reset_client_for_testing()
    token = _get_token(client)
    response = client.post(
        "/api/copilot/chat", json={"message": "hello"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_chat_endpoint_requires_authentication(client):
    response = client.post("/api/copilot/chat", json={"message": "hello"})
    assert response.status_code == 401


# --- Morning Brief: same guardrails/plumbing as chat(), just a fixed prompt ---


def test_morning_brief_raises_copilot_unavailable_when_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent.reset_client_for_testing()
    with pytest.raises(agent.CopilotUnavailableError):
        agent.generate_morning_brief(tenant_id="telco")


def test_morning_brief_endpoint_returns_503_when_key_missing(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent.reset_client_for_testing()
    token = _get_token(client)
    response = client.get("/api/copilot/morning-brief", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_morning_brief_endpoint_requires_authentication(client):
    response = client.get("/api/copilot/morning-brief")
    assert response.status_code == 401


def test_morning_brief_delegates_to_chat_with_fixed_prompt_and_tenant(monkeypatch):
    captured = {}

    def fake_chat(message, tenant_id, db=None):
        captured["message"] = message
        captured["tenant_id"] = tenant_id
        return {"response": "fake brief", "sources": [], "tool_outputs": []}

    monkeypatch.setattr(agent, "chat", fake_chat)
    result = agent.generate_morning_brief(tenant_id="telco")

    assert captured["message"] == agent.MORNING_BRIEF_PROMPT
    assert captured["tenant_id"] == "telco"
    assert result == {"response": "fake brief", "sources": [], "tool_outputs": []}


# --- Tool declaration schemas must never expose tenant_id to the model ---


def test_no_tool_declaration_exposes_tenant_id():
    for declaration in agent._TOOL_DECLARATIONS:
        schema = declaration.parameters_json_schema or {}
        properties = schema.get("properties", {})
        assert "tenant_id" not in properties, f"{declaration.name} must never let the model choose a tenant"


def test_tool_declaration_names_match_real_tools_module():
    from src.copilot import tools as tools_module

    declared_names = {declaration.name for declaration in agent._TOOL_DECLARATIONS}
    real_tool_names = {
        "get_business_impact_summary",
        "get_top_opportunities",
        "get_customer_detail",
        "run_budget_optimization",
        "run_scenario",
        "get_segment_summary",
        "get_alerts",
        "get_survival_likelihood",
    }
    assert declared_names == real_tool_names
    for name in real_tool_names:
        assert hasattr(tools_module, name)


# --- Dispatch must always use the server-side tenant_id, never a model-supplied one ---


def test_execute_tool_injects_server_tenant_id_ignoring_model_supplied_one(monkeypatch):
    captured = {}

    def fake_get_business_impact_summary(tenant_id, db=None):
        captured["tenant_id"] = tenant_id
        return {"source": "fake"}

    monkeypatch.setattr(agent.tools, "get_business_impact_summary", fake_get_business_impact_summary)

    # Simulate a model that hallucinated a tenant_id into its function-call
    # arguments anyway - the real, authenticated tenant_id must win.
    result = agent._execute_tool("get_business_impact_summary", {"tenant_id": "banking"}, tenant_id="telco")

    assert captured["tenant_id"] == "telco"
    assert result == {"source": "fake"}


def test_execute_tool_routes_get_customer_detail_with_args_and_tenant(monkeypatch):
    captured = {}

    def fake_get_customer_detail(customer_id, tenant_id, db=None):
        captured["customer_id"] = customer_id
        captured["tenant_id"] = tenant_id
        return {"source": "fake"}

    monkeypatch.setattr(agent.tools, "get_customer_detail", fake_get_customer_detail)
    agent._execute_tool("get_customer_detail", {"customer_id": "7590-VHVEG"}, tenant_id="telco")

    assert captured == {"customer_id": "7590-VHVEG", "tenant_id": "telco"}


def test_execute_tool_routes_run_budget_optimization_with_defaults(monkeypatch):
    captured = {}

    def fake_run_budget_optimization(tenant_id, budget, cost_override, db=None):
        captured.update(tenant_id=tenant_id, budget=budget, cost_override=cost_override)
        return {"source": "fake"}

    monkeypatch.setattr(agent.tools, "run_budget_optimization", fake_run_budget_optimization)
    agent._execute_tool("run_budget_optimization", {"budget": 5000}, tenant_id="telco")

    assert captured == {"tenant_id": "telco", "budget": 5000.0, "cost_override": None}


def test_execute_tool_unknown_tool_name_returns_error_not_fabrication():
    result = agent._execute_tool("not_a_real_tool", {}, tenant_id="telco")
    assert "error" in result
