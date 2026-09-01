"""SQLAlchemy ORM schema for the tenant-scoped persistence layer.

Every table carries tenant_id. There is no cross-tenant relationship or
shared row of any kind - isolation is enforced by always filtering on
tenant_id (see database.db.get_tenant_scoped_query), never by table
structure alone.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    raw_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    scenario_name: Mapped[str] = mapped_column(String, nullable=False)
    scenario_type: Mapped[str] = mapped_column(String, nullable=False)
    params_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    results_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


class RecommendationLog(Base):
    __tablename__ = "recommendation_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    recommended_action: Mapped[str] = mapped_column(String, nullable=False)
    triggered_by_json: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


class CustomerTracking(Base):
    """Manual operational fields a user fills in against a customer -
    assigned_manager/status/call_scheduled_date are never AI-generated or
    derived from any model; they only ever hold whatever a human typed in
    via PUT /api/tracking/{customer_id}."""

    __tablename__ = "customer_tracking"
    __table_args__ = (UniqueConstraint("tenant_id", "customer_id", name="uq_customer_tracking_tenant_customer"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    assigned_manager: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="not_started")
    call_scheduled_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class ApiKey(Base):
    """A system-to-system credential, scoped to one tenant. Only key_hash is
    ever stored - the raw key is generated, returned once in the create
    response, and never retrievable again. masked_key is computed once at
    creation time (prefix + last 4 chars of the raw key) purely for display
    in the key-management UI/list endpoint."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    key_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    masked_key: Mapped[str] = mapped_column(String, nullable=False)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class OnboardingUpload(Base):
    """A validated, security-checked CSV upload staged between the upload
    and validate steps of self-service data onboarding (Stage 1). Storing
    the sanitized CSV text in the DB (rather than a filesystem path) keeps
    this tenant-scoped and avoids introducing a new storage dependency for
    a feature that doesn't train anything yet - see docs/ADDING_A_TENANT.md
    for what a REAL new-tenant onboarding (training included) still
    requires beyond this."""

    __tablename__ = "onboarding_uploads"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    columns_json: Mapped[list] = mapped_column(JSON, nullable=False)
    csv_text: Mapped[str] = mapped_column(String, nullable=False)
    # {csv_column_name: role}, the USER-CONFIRMED mapping from the mapping
    # step - set by POST /api/onboarding/validate, None until then. This is
    # what src/models/tenant_training.py reads to know which column is the
    # id/target/revenue/features when a training job later runs against
    # this upload - never re-derived from suggest_mapping()'s guesses.
    column_mapping_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


class Company(Base):
    """A self-registered company (POST /api/companies/register) - the
    authoritative record that a tenant_id was created via self-service
    signup, distinct from config.yaml's statically-defined, pre-trained
    tenants (telco/banking). Doubles as this tenant's DYNAMIC profile once
    a training job (api/training.py) succeeds - model_dir/data_path/
    feature_flags_json are all None/empty until then, so every
    feature_flags check naturally keeps returning "not yet trained for
    this tenant" (via src/tenant_registry.py's Company-backed fallback for
    any tenant_id not in config.yaml's static profiles) until training
    actually completes with a sane result."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    # Set together, only once a training job succeeds - model_dir/data_path
    # point at what src/models/train.py's train_model() actually wrote (see
    # src/models/tenant_training.py), never guessed or pre-created at
    # registration time.
    model_dir: Mapped[str | None] = mapped_column(String, nullable=True)
    data_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Mirrors config.yaml tenant profiles' feature_flags shape
    # ({flag_name: bool}) so src/tenant_registry.py can treat both sources
    # uniformly. Only ever set to a non-empty dict by a training job that
    # passed its sanity check (see api/training.py's SANE_ROC_AUC bounds) -
    # a job that succeeds with a suspicious result leaves this untouched
    # (still not-enabled) rather than auto-flipping anything.
    feature_flags_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class OnboardingStatus(Base):
    """Tracks which onboarding step a self-registered company is on, so the
    frontend wizard can resume a partially-completed flow instead of
    starting over. Only ever created for tenants that went through
    POST /api/companies/register - see database/onboarding_status.py's
    advance_status(), which is a no-op for any tenant without a row here
    (e.g. the pre-existing telco/banking demo tenants)."""

    __tablename__ = "onboarding_status"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    step: Mapped[str] = mapped_column(String, nullable=False, default="registered")
    last_upload_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class TrainingJob(Base):
    """Async background-job lifecycle tracking for tenant model training
    (Stage 2a - see api/training.py's module docstring). status moves
    forward only, one of queued -> running -> (succeeded | failed).
    started_at/completed_at/error_message/result_metadata_json are all
    None until the background thread actually reaches that point - never
    backfilled or guessed."""

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    result_metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
