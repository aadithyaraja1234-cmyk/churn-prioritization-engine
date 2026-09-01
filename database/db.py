"""Engine/session setup and the single enforcement point for tenant isolation.

Every future endpoint or script that reads customers/predictions must go
through get_tenant_scoped_query() rather than querying the ORM models
directly - that is what guarantees a request for one tenant can never see
another tenant's rows, even though they live in the same physical table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Type, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.orm import Query, Session, sessionmaker

from database.models import Base

# DATABASE_PATH env var override, for a real deployment where the DB needs
# to live on a mounted volume outside the source tree (e.g. Docker - see
# docker-compose.yml's `backend_data` volume) rather than inside the app
# directory itself, which a container rebuild would otherwise wipe. Unset
# (the default - every existing local-dev/test invocation) reproduces the
# exact previous path, so this is a no-op everywhere except where the env
# var is deliberately set.
DB_PATH = Path(os.environ.get("DATABASE_PATH") or (Path(__file__).resolve().parent / "app.db"))
# check_same_thread=False: training_jobs' background thread (api/training.py)
# opens its own session against this same engine from a thread that isn't
# the one FastAPI's request/response cycle runs on - the standard, safe
# SQLite+FastAPI pattern for that (each session still gets its own
# connection; this only lifts sqlite3's default same-thread-as-creator
# restriction on a pooled connection being handed to a different thread).
ENGINE = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
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
