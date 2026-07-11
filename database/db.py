"""Engine/session setup and the single enforcement point for tenant isolation.

Every future endpoint or script that reads customers/predictions must go
through get_tenant_scoped_query() rather than querying the ORM models
directly - that is what guarantees a request for one tenant can never see
another tenant's rows, even though they live in the same physical table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Type, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.orm import Query, Session, sessionmaker

from database.models import Base

DB_PATH = Path(__file__).resolve().parent / "app.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=ENGINE)

ModelT = TypeVar("ModelT")


def init_db() -> None:
    Base.metadata.create_all(ENGINE)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_tenant_scoped_query(db: Session, model: Type[ModelT], tenant_id: str) -> Query:
    """Return a query for `model` pre-filtered to tenant_id.

    `model` must declare a tenant_id column (User, Customer, Prediction all
    do). Callers must never call db.query(model) directly for these tables.
    """
    return db.query(model).filter(model.tenant_id == tenant_id)
