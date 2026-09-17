"""Company self-registration: the first-impression signup flow for a new
company, distinct from api/auth.py's /auth/register (which adds a user to
an EXISTING tenant - used by internal demo accounts). This creates a
brand-new tenant_id, its first (admin) user, and an onboarding-status row
to track progress through the data-upload wizard - it does NOT train a
model or touch config.yaml's tenant profiles.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import CurrentUser, create_access_token, get_current_user, hash_password
from api.rate_limit import enforce_ip_rate_limit
from database.db import get_db
from database.models import (
    ApiKey,
    Company,
    Customer,
    CustomerTracking,
    OnboardingStatus,
    OnboardingUpload,
    Prediction,
    RecommendationLog,
    Scenario,
    User,
)
from database.onboarding_status import create_status, get_status
from src.tenant_registry import get_tenant_profile, load_tenant_profiles
from src.validation.credentials import (
    slugify_company_name,
    validate_company_name,
    validate_email_format,
    validate_password_strength,
)

router = APIRouter(prefix="/api/companies", tags=["Company Registration"])

# Same reasoning/value as api/auth.py's REGISTER_RATE_LIMIT_PER_MINUTE -
# guards this, the real self-service signup endpoint, against registration
# spam/account-enumeration abuse. Kept as its own constant rather than
# imported from auth.py since this is conceptually a distinct endpoint,
# not a call-through to it.
COMPANY_REGISTER_RATE_LIMIT_PER_MINUTE = 5

_CONFIG_TENANT_PROFILES = load_tenant_profiles()

ROOT = Path(__file__).resolve().parent.parent


class CompanyRegisterRequest(BaseModel):
    company_name: str
    email: str
    password: str


class CompanyRegisterResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    company_name: str
    onboarding_step: str


@router.post(
    "/register",
    response_model=CompanyRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new company and its first (admin) user",
    description="Creates a brand-new tenant, its first admin user, and an onboarding-status "
    "record - this is self-service company signup, not adding a user to an existing tenant. "
    "Validates every field and returns ALL field-level "
    "errors at once as `{\"errors\": {\"company_name\": ..., \"email\": ..., \"password\": "
    "...}}` (400), never a single generic message. Does not train a model or touch any "
    "existing tenant - the new tenant starts with every advanced feature reporting "
    "'not yet trained for this tenant' until real data is uploaded and validated.",
)
def register_company(
    http_request: Request, request: CompanyRegisterRequest, db: Session = Depends(get_db)
) -> CompanyRegisterResponse:
    enforce_ip_rate_limit(http_request, scope="company_register", limit_per_minute=COMPANY_REGISTER_RATE_LIMIT_PER_MINUTE)

    errors: dict[str, str] = {}

    name_error = validate_company_name(request.company_name)
    if name_error:
        errors["company_name"] = name_error

    email_error = validate_email_format(request.email)
    if email_error:
        errors["email"] = email_error

    password_error = validate_password_strength(request.password)
    if password_error:
        errors["password"] = password_error

    if "email" not in errors and db.query(User).filter(User.email == request.email).first() is not None:
        errors["email"] = "This email is already registered."

    tenant_id = slugify_company_name(request.company_name) if "company_name" not in errors else None
    if tenant_id is not None and db.query(Company).filter(Company.tenant_id == tenant_id).first() is not None:
        errors["company_name"] = "A company with a very similar name is already registered - please choose a different name."

    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    company = Company(tenant_id=tenant_id, company_name=request.company_name.strip())
    db.add(company)

    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        tenant_id=tenant_id,
        role="admin",
    )
    db.add(user)
    db.commit()

    status_row = create_status(db, tenant_id, step="registered")

    token = create_access_token(email=user.email, tenant_id=tenant_id, role="admin")

    return CompanyRegisterResponse(
        access_token=token,
        tenant_id=tenant_id,
        company_name=company.company_name,
        onboarding_step=status_row.step,
    )


@router.get(
    "/onboarding-status",
    summary="Get which onboarding step your company is on",
    description="Returns `step` (registered / data_uploaded / data_validated / ready) and, if "
    "applicable, `last_upload_id` so the frontend wizard can resume a partially-completed "
    "onboarding instead of starting over. Pre-existing, already-trained tenants (e.g. the demo "
    "Telco/Banking tenants) report `ready` even without ever going through this registration "
    "flow. Returns 404 only if your tenant is neither a self-registered company nor a "
    "pre-existing trained tenant.",
)
def onboarding_status(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    row = get_status(db, current_user.tenant_id)
    if row is not None:
        return {
            "tenant_id": row.tenant_id,
            "step": row.step,
            "last_upload_id": row.last_upload_id,
            "updated_at": row.updated_at.isoformat(),
        }

    if get_tenant_profile(current_user.tenant_id, _CONFIG_TENANT_PROFILES) is not None:
        return {"tenant_id": current_user.tenant_id, "step": "ready", "last_upload_id": None, "updated_at": None}

    raise HTTPException(status_code=404, detail="Onboarding status not found for this tenant.")


def _tenant_model_dir(tenant_id: str) -> Path:
    return ROOT / "models" / tenant_id


def _tenant_upload_path(tenant_id: str) -> Path:
    return ROOT / "data" / "tenant_uploads" / f"{tenant_id}.csv"


@router.get(
    "/export",
    summary="Export every stored record for your tenant",
    description="Returns everything this product has stored for your tenant as a single JSON "
    "document: your company record, users (email/role only - never a password hash or API key "
    "secret), uploaded customer data, predictions, scenarios, recommendation logs, manual "
    "tracking fields, onboarding uploads, and API key metadata (masked key only, never the raw "
    "key or its hash). For fulfilling a data-portability/export request. Available to any "
    "authenticated user of the tenant.",
)
def export_company_data(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    tenant_id = current_user.tenant_id
    company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
    status_row = get_status(db, tenant_id)

    return {
        "tenant_id": tenant_id,
        "company": (
            {
                "company_name": company.company_name,
                "created_at": company.created_at.isoformat(),
                "feature_flags": company.feature_flags_json,
            }
            if company is not None
            else None
        ),
        "onboarding_status": (
            {"step": status_row.step, "last_upload_id": status_row.last_upload_id}
            if status_row is not None
            else None
        ),
        "users": [
            {"email": row.email, "role": row.role, "created_at": row.created_at.isoformat()}
            for row in db.query(User).filter(User.tenant_id == tenant_id).all()
        ],
        "customers": [
            {"customer_id": row.customer_id, "raw_features": row.raw_features, "created_at": row.created_at.isoformat()}
            for row in db.query(Customer).filter(Customer.tenant_id == tenant_id).all()
        ],
        "predictions": [
            {
                "customer_id": row.customer_id,
                "model_version": row.model_version,
                "churn_probability": row.churn_probability,
                "predicted_at": row.predicted_at.isoformat(),
            }
            for row in db.query(Prediction).filter(Prediction.tenant_id == tenant_id).all()
        ],
        "scenarios": [
            {
                "scenario_name": row.scenario_name,
                "scenario_type": row.scenario_type,
                "params": row.params_json,
                "results": row.results_json,
                "created_at": row.created_at.isoformat(),
            }
            for row in db.query(Scenario).filter(Scenario.tenant_id == tenant_id).all()
        ],
        "recommendation_log": [
            {
                "customer_id": row.customer_id,
                "recommended_action": row.recommended_action,
                "triggered_by": row.triggered_by_json,
                "created_at": row.created_at.isoformat(),
            }
            for row in db.query(RecommendationLog).filter(RecommendationLog.tenant_id == tenant_id).all()
        ],
        "customer_tracking": [
            {
                "customer_id": row.customer_id,
                "assigned_manager": row.assigned_manager,
                "status": row.status,
                "call_scheduled_date": row.call_scheduled_date.isoformat() if row.call_scheduled_date else None,
                "updated_at": row.updated_at.isoformat(),
            }
            for row in db.query(CustomerTracking).filter(CustomerTracking.tenant_id == tenant_id).all()
        ],
        "onboarding_uploads": [
            {
                "filename": row.filename,
                "row_count": row.row_count,
                "columns": row.columns_json,
                "column_mapping": row.column_mapping_json,
                "csv_text": row.csv_text,
                "created_at": row.created_at.isoformat(),
            }
            for row in db.query(OnboardingUpload).filter(OnboardingUpload.tenant_id == tenant_id).all()
        ],
        # Masked key + metadata only - key_hash is never included, and the raw key was
        # never retrievable after creation in the first place.
        "api_keys": [
            {
                "name": row.name,
                "masked_key": row.masked_key,
                "rate_limit_per_minute": row.rate_limit_per_minute,
                "created_at": row.created_at.isoformat(),
                "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
                "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
            }
            for row in db.query(ApiKey).filter(ApiKey.tenant_id == tenant_id).all()
        ],
    }


@router.delete(
    "/account",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Permanently delete your company account and all associated data",
    description="Irreversibly deletes every stored record for your tenant (users, uploaded "
    "customer data, predictions, scenarios, recommendation logs, tracking fields, onboarding "
    "uploads, API keys) plus any trained model artifacts and uploaded CSV on disk. Restricted "
    "to an admin user of a self-registered tenant - the built-in reference tenants (Telco/"
    "Banking) have no Company row and cannot be deleted this way. Does not invalidate any JWT "
    "issued before deletion (it simply expires at its normal 60-minute lifetime, same as any "
    "other JWT in this system) - a genuinely revocable-token scheme is a separate, larger change.",
)
def delete_company_account(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> None:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only an admin can delete the company account.")

    tenant_id = current_user.tenant_id
    company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
    if company is None:
        raise HTTPException(
            status_code=404,
            detail="No self-registered company found for this tenant - built-in reference "
            "tenants cannot be deleted through this endpoint.",
        )

    for model in (
        User,
        Customer,
        Prediction,
        Scenario,
        RecommendationLog,
        CustomerTracking,
        ApiKey,
        OnboardingUpload,
    ):
        db.query(model).filter(model.tenant_id == tenant_id).delete(synchronize_session=False)
    db.query(OnboardingStatus).filter(OnboardingStatus.tenant_id == tenant_id).delete(synchronize_session=False)
    db.delete(company)
    db.commit()

    model_dir = _tenant_model_dir(tenant_id)
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
    upload_path = _tenant_upload_path(tenant_id)
    if upload_path.exists():
        upload_path.unlink()
