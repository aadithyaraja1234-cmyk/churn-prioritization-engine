"""Company self-registration: the first-impression signup flow for a new
company, distinct from api/auth.py's /auth/register (which adds a user to
an EXISTING tenant - used by internal demo accounts). This creates a
brand-new tenant_id, its first (admin) user, and an onboarding-status row
to track progress through the data-upload wizard - it does NOT train a
model or touch config.yaml's tenant profiles.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import CurrentUser, create_access_token, get_current_user, hash_password
from database.db import get_db
from database.models import Company, User
from database.onboarding_status import create_status, get_status
from src.tenant_registry import get_tenant_profile, load_tenant_profiles
from src.validation.credentials import (
    slugify_company_name,
    validate_company_name,
    validate_email_format,
    validate_password_strength,
)

router = APIRouter(prefix="/api/companies", tags=["Company Registration"])

_CONFIG_TENANT_PROFILES = load_tenant_profiles()


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
def register_company(request: CompanyRegisterRequest, db: Session = Depends(get_db)) -> CompanyRegisterResponse:
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
