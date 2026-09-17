"""Tests for the forgot-password/reset-password flow (api/auth.py's
/forgot-password, /reset-password) - the account-recovery path that
didn't exist at all before this. src/notifications/email.py is mocked at
its own send_password_reset_email() boundary (same style as
tests/test_copilot_agent.py mocking agent.chat rather than the real
Gemini SDK) so these never make a real network call to Resend, and so the
raw reset token - which never appears in any HTTP response, by design -
can still be captured for the next step in a test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import auth as auth_module
from api.main import app
from database.db import get_db
from database.models import Base, User


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


@pytest.fixture()
def captured_reset_links(monkeypatch):
    """Replaces the real send - captures (to_email, reset_link) pairs
    instead, so a test can pull the token out of the link without ever
    going through an HTTP response (which must never carry it - see
    api/auth.py's forgot_password() docstring)."""
    calls = []
    monkeypatch.setattr(auth_module, "send_password_reset_email", lambda *, to_email, reset_link: calls.append((to_email, reset_link)))
    return calls


def _register(client, email="reset-me@example.com", password="original-pw-123"):
    return client.post(
        "/auth/register",
        json={"email": email, "password": password, "tenant_id": "telco", "role": "analyst"},
    )


def _extract_token(reset_link: str) -> str:
    return reset_link.split("token=", 1)[1]


def test_forgot_password_for_real_user_sends_email_and_returns_generic_message(client, captured_reset_links):
    _register(client, email="real-user@example.com")

    response = client.post("/auth/forgot-password", json={"email": "real-user@example.com"})
    assert response.status_code == 200
    assert "if that email is registered" in response.json()["message"].lower()

    assert len(captured_reset_links) == 1
    to_email, reset_link = captured_reset_links[0]
    assert to_email == "real-user@example.com"
    assert "token=" in reset_link


def test_forgot_password_for_unknown_email_returns_same_generic_message_and_sends_nothing(client, captured_reset_links):
    """Account-enumeration defense: the response must be identical to the
    real-user case, and no email attempt should even be made."""
    known = client.post("/auth/forgot-password", json={"email": "nobody-registered@example.com"})
    assert known.status_code == 200
    assert "if that email is registered" in known.json()["message"].lower()
    assert captured_reset_links == []


def test_reset_password_with_valid_token_logs_in_with_new_password(client, captured_reset_links):
    _register(client, email="reset-me@example.com", password="original-pw-123")
    client.post("/auth/forgot-password", json={"email": "reset-me@example.com"})
    token = _extract_token(captured_reset_links[0][1])

    reset_response = client.post(
        "/auth/reset-password", json={"token": token, "new_password": "brand-new-pw-456"}
    )
    assert reset_response.status_code == 200, reset_response.text
    assert reset_response.json()["access_token"]

    # Old password must no longer work...
    old_login = client.post("/auth/login", json={"email": "reset-me@example.com", "password": "original-pw-123"})
    assert old_login.status_code == 401
    # ...new one must.
    new_login = client.post("/auth/login", json={"email": "reset-me@example.com", "password": "brand-new-pw-456"})
    assert new_login.status_code == 200


def test_reset_password_token_is_single_use(client, captured_reset_links):
    _register(client, email="reset-me@example.com")
    client.post("/auth/forgot-password", json={"email": "reset-me@example.com"})
    token = _extract_token(captured_reset_links[0][1])

    first = client.post("/auth/reset-password", json={"token": token, "new_password": "brand-new-pw-456"})
    assert first.status_code == 200

    second = client.post("/auth/reset-password", json={"token": token, "new_password": "another-pw-789"})
    assert second.status_code == 400
    assert "invalid or has expired" in second.json()["detail"].lower()


def test_reset_password_rejects_unknown_token(client):
    response = client.post("/auth/reset-password", json={"token": "not-a-real-token", "new_password": "brand-new-pw-456"})
    assert response.status_code == 400
    assert "invalid or has expired" in response.json()["detail"].lower()


def test_reset_password_rejects_weak_new_password(client, captured_reset_links):
    _register(client, email="reset-me@example.com")
    client.post("/auth/forgot-password", json={"email": "reset-me@example.com"})
    token = _extract_token(captured_reset_links[0][1])

    response = client.post("/auth/reset-password", json={"token": token, "new_password": "weak"})
    assert response.status_code == 400
    # The token must still be usable after a rejected-for-weak-password
    # attempt - only a SUCCESSFUL reset consumes it.
    retry = client.post("/auth/reset-password", json={"token": token, "new_password": "a-strong-pw-789"})
    assert retry.status_code == 200


def test_requesting_a_new_reset_invalidates_the_previous_token(client, captured_reset_links):
    _register(client, email="reset-me@example.com")
    client.post("/auth/forgot-password", json={"email": "reset-me@example.com"})
    first_token = _extract_token(captured_reset_links[0][1])

    client.post("/auth/forgot-password", json={"email": "reset-me@example.com"})
    second_token = _extract_token(captured_reset_links[1][1])
    assert second_token != first_token

    stale_attempt = client.post("/auth/reset-password", json={"token": first_token, "new_password": "brand-new-pw-456"})
    assert stale_attempt.status_code == 400

    fresh_attempt = client.post("/auth/reset-password", json={"token": second_token, "new_password": "brand-new-pw-456"})
    assert fresh_attempt.status_code == 200
