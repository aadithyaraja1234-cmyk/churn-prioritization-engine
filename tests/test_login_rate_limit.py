"""Tests for api/rate_limit.py and its wiring into /auth/login, /auth/
register, and /api/companies/register - the brute-force/registration-spam
protection these three unauthenticated endpoints had none of before.
Mirrors tests/test_api_keys.py's own rate-limit tests (429 once exceeded,
independent buckets) for this IP-keyed limiter instead of the per-key one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base


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


def test_login_returns_429_once_rate_limit_exceeded(client):
    client.post(
        "/auth/register",
        json={"email": "brute@example.com", "password": "s3cret-pw", "tenant_id": "telco", "role": "analyst"},
    )

    # LOGIN_RATE_LIMIT_PER_MINUTE=10 - the 11th attempt in this window must
    # 429, regardless of whether the password guess is right or wrong (the
    # limiter runs before credential verification).
    for _ in range(10):
        response = client.post("/auth/login", json={"email": "brute@example.com", "password": "wrong-guess"})
        assert response.status_code == 401

    limited = client.post("/auth/login", json={"email": "brute@example.com", "password": "wrong-guess"})
    assert limited.status_code == 429
    assert "too many attempts" in limited.json()["detail"].lower()


def test_login_rate_limit_does_not_block_a_different_endpoint(client):
    """A caller who exhausted the LOGIN bucket must not also be blocked
    from /auth/register - scope="login" vs scope="register" are separate
    buckets for the same IP (see api/rate_limit.py's enforce_ip_rate_limit
    `scope` parameter)."""
    for _ in range(10):
        client.post("/auth/login", json={"email": "nobody@example.com", "password": "x"})
    assert client.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "x"}
    ).status_code == 429

    register_response = client.post(
        "/auth/register",
        json={"email": "still-works@example.com", "password": "s3cret-pw", "tenant_id": "telco", "role": "analyst"},
    )
    assert register_response.status_code == 201


def test_company_register_returns_429_once_rate_limit_exceeded(client):
    # COMPANY_REGISTER_RATE_LIMIT_PER_MINUTE=5.
    for i in range(5):
        response = client.post(
            "/api/companies/register",
            json={"company_name": f"Spam Co {i}", "email": f"spam{i}@example.com", "password": "s3cret-pw1"},
        )
        assert response.status_code == 201, response.text

    limited = client.post(
        "/api/companies/register",
        json={"company_name": "Spam Co Six", "email": "spam6@example.com", "password": "s3cret-pw1"},
    )
    assert limited.status_code == 429


def test_successful_login_still_works_under_the_limit(client):
    client.post(
        "/auth/register",
        json={"email": "normal-user@example.com", "password": "s3cret-pw", "tenant_id": "telco", "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": "normal-user@example.com", "password": "s3cret-pw"})
    assert response.status_code == 200
    assert response.json()["access_token"]
