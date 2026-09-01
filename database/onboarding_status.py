"""Tracks which self-service onboarding step a company is on: registered ->
data_uploaded -> data_validated -> ready. Only ever exists for tenants
created via POST /api/companies/register - advance_status() is a
deliberate no-op for any tenant_id without a row (e.g. the pre-existing
telco/banking demo tenants using /api/onboarding/* endpoints for a
preview, per Stage 1's scope), so this never affects them.

Advancement is monotonic - calling advance_status() with an earlier step
than the tenant is already on is a no-op, so a user re-uploading data
after already reaching "ready" doesn't get bumped backward by accident.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from database.models import OnboardingStatus

STEP_ORDER = ["registered", "data_uploaded", "data_validated", "ready"]


def create_status(db: Session, tenant_id: str, step: str = "registered") -> OnboardingStatus:
    row = OnboardingStatus(tenant_id=tenant_id, step=step)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_status(db: Session, tenant_id: str) -> OnboardingStatus | None:
    return db.query(OnboardingStatus).filter(OnboardingStatus.tenant_id == tenant_id).first()


def advance_status(db: Session, tenant_id: str, step: str, upload_id: int | None = None) -> OnboardingStatus | None:
    row = get_status(db, tenant_id)
    if row is None:
        return None  # not a self-registered company being tracked - no-op

    if STEP_ORDER.index(step) > STEP_ORDER.index(row.step):
        row.step = step
    if upload_id is not None:
        row.last_upload_id = upload_id
    db.commit()
    db.refresh(row)
    return row
