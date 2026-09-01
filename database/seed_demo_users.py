"""Idempotent seed script: creates two permanent demo accounts (one per
tenant) if they don't already exist. Safe to run repeatedly - re-running
never duplicates or resets an existing demo user's password.
"""

from __future__ import annotations

from api.auth import hash_password
from database.db import SessionLocal, init_db
from database.models import User

DEMO_USERS = [
    {"email": "demo-telco@churn-engine.local", "password": "DemoPass123", "tenant_id": "telco", "role": "admin"},
    {"email": "demo-banking@churn-engine.local", "password": "DemoPass123", "tenant_id": "banking", "role": "admin"},
]


def seed_demo_users() -> list[str]:
    init_db()
    session = SessionLocal()
    created = []
    try:
        for demo_user in DEMO_USERS:
            existing = session.query(User).filter(User.email == demo_user["email"]).first()
            if existing is not None:
                continue
            session.add(
                User(
                    email=demo_user["email"],
                    hashed_password=hash_password(demo_user["password"]),
                    tenant_id=demo_user["tenant_id"],
                    role=demo_user["role"],
                )
            )
            created.append(demo_user["email"])
        session.commit()
    finally:
        session.close()
    return created


if __name__ == "__main__":
    created = seed_demo_users()
    if created:
        print("Created:", ", ".join(created))
    else:
        print("No new users created - both demo accounts already exist.")
