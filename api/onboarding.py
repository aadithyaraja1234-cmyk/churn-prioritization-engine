"""Self-service data onboarding, Stage 1: secure upload, guided column
mapping, and data-sufficiency validation only. Deliberately does NOT train
a model or create a new tenant/company - see this module's endpoints'
descriptions and docs/ADDING_A_TENANT.md for what a real new-tenant
onboarding still requires beyond this.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import CurrentUser, get_current_user
from database.db import get_db, get_tenant_scoped_query
from database.models import OnboardingUpload
from database.onboarding_status import advance_status
from src.onboarding.ai_mapping import refine_mapping_with_ai
from src.onboarding.schema_fields import VALID_ROLES, suggest_duration_leakage_column, suggest_mapping
from src.onboarding.upload_security import MAX_FILE_SIZE_BYTES, MAX_ROWS, UploadRejected, read_and_validate_upload
from src.onboarding.validation import validate_upload

router = APIRouter(prefix="/api/onboarding", tags=["Data Onboarding"])

# A fast-path rejection using the Content-Length header, BEFORE we ever
# trigger multipart body parsing - catches the common case of a
# wildly-oversized upload without reading a single byte of it. This is a
# coarse pre-check only (Content-Length can be absent for chunked transfer
# encoding, or simply wrong); the authoritative enforcement is the
# streamed, chunk-by-chunk byte count in read_and_validate_upload(), which
# never trusts any header and aborts the instant the real cumulative byte
# count crosses the limit.
CONTENT_LENGTH_OVERHEAD_ALLOWANCE = 64 * 1024


class ColumnMappingRequest(BaseModel):
    upload_id: int
    column_mapping: dict[str, str]


@router.post(
    "/upload",
    summary="Upload a CSV file for self-service data onboarding",
    description="Accepts a multipart CSV file upload (field name `file`), enforces a 10MB / "
    "50,000-row hard limit (rejected via a fast Content-Length pre-check and an authoritative "
    "streamed byte-count check, never by fully loading an oversized file into memory), verifies "
    "the content is genuinely parseable CSV text (never trusting the .csv extension or declared "
    "content-type), and rejects any cell value that looks like a spreadsheet formula or contains "
    "a non-printable control character (CSV-injection / terminal-escape-sequence-injection "
    "defense: reject-and-explain, not silent sanitization). This does NOT scan for classic "
    "file-based malware (an executable/macro document/etc.) - not needed for a CSV-only, "
    "rendered-as-plain-text pipeline like this one; see src/onboarding/upload_security.py's "
    "module docstring for the full reasoning. On success, returns the parsed columns, a small "
    "preview, and a "
    "suggested column mapping (name-similarity first, then an AI-assisted second pass via Gemini "
    "for whichever columns that first pass couldn't confidently match - degrades silently to "
    "name-similarity only if GEMINI_API_KEY isn't set; never auto-applied either way, confirm or "
    "change every field before calling /validate).",
)
async def upload_csv(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > MAX_FILE_SIZE_BYTES + CONTENT_LENGTH_OVERHEAD_ALLOWANCE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB upload limit.",
            )

    form = await request.form()
    upload_file = form.get("file")
    if upload_file is None or not hasattr(upload_file, "read"):
        raise HTTPException(status_code=400, detail="No file uploaded - expected a multipart field named 'file'.")

    try:
        csv_text = await read_and_validate_upload(upload_file)
    except UploadRejected as exc:
        raise HTTPException(status_code=400, detail=exc.reason)

    df = pd.read_csv(io.StringIO(csv_text))
    columns = [str(c) for c in df.columns]
    preview_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")
    suggested_mapping = suggest_mapping(columns)
    suggested_mapping = suggest_duration_leakage_column(df, suggested_mapping)
    suggested_mapping = refine_mapping_with_ai(columns, preview_rows, suggested_mapping)

    row = OnboardingUpload(
        tenant_id=current_user.tenant_id,
        filename=getattr(upload_file, "filename", None) or "upload.csv",
        row_count=len(df),
        columns_json=columns,
        csv_text=csv_text,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # No-op for tenants that never registered via POST /api/companies/register
    # (e.g. the demo telco/banking tenants using this endpoint for a preview).
    advance_status(db, current_user.tenant_id, "data_uploaded", upload_id=row.id)

    return {
        "upload_id": row.id,
        "filename": row.filename,
        "row_count": row.row_count,
        "columns": columns,
        "preview_rows": preview_rows,
        "suggested_mapping": suggested_mapping,
        "valid_roles": list(VALID_ROLES),
    }


@router.get(
    "/upload/{upload_id}",
    summary="Re-fetch a previously uploaded file's parsed preview and mapping suggestion",
    description="Returns the same shape as POST /upload's response for a file already "
    "uploaded - used by the onboarding wizard to resume where a user left off (e.g. after "
    "navigating away and back) without re-uploading. Returns 404 if `upload_id` doesn't belong "
    "to your tenant.",
)
def get_upload(
    upload_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = (
        get_tenant_scoped_query(db, OnboardingUpload, tenant_id=current_user.tenant_id)
        .filter(OnboardingUpload.id == upload_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    df = pd.read_csv(io.StringIO(row.csv_text))
    columns = [str(c) for c in df.columns]
    preview_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")
    suggested_mapping = suggest_duration_leakage_column(df, suggest_mapping(columns))
    suggested_mapping = refine_mapping_with_ai(columns, preview_rows, suggested_mapping)

    return {
        "upload_id": row.id,
        "filename": row.filename,
        "row_count": row.row_count,
        "columns": columns,
        "preview_rows": preview_rows,
        "suggested_mapping": suggested_mapping,
        "valid_roles": list(VALID_ROLES),
    }


@router.post(
    "/validate",
    summary="Validate a previously uploaded file against a confirmed column mapping",
    description="Runs structural checks (a customer-id column mapped and unique, a target "
    "column mapped with at least two outcome classes) plus a graded data-sufficiency "
    "assessment (row count and usable-feature count against this system's own real Telco "
    "reference-model thresholds) against `upload_id`'s file, using the CALLER-CONFIRMED "
    "`column_mapping` ({csv_column_name: role}, role one of customer_id/target/revenue/"
    "feature/duration/clv/duration_leakage_column/ignore) - never a suggestion applied "
    "automatically. Structural check failures "
    "block proceeding (`can_proceed: false`); data-sufficiency issues are a warning only "
    "(`data_sufficiency.requires_acknowledgment: true`) with specific, actionable guidance, not "
    "a hard block - the caller may still proceed with an explicit 'results may be less "
    "reliable' acknowledgment. Returns 404 if `upload_id` doesn't belong to your tenant.",
)
def validate_upload_endpoint(
    request: ColumnMappingRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = (
        get_tenant_scoped_query(db, OnboardingUpload, tenant_id=current_user.tenant_id)
        .filter(OnboardingUpload.id == request.upload_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    df = pd.read_csv(io.StringIO(row.csv_text))
    try:
        result = validate_upload(df, request.column_mapping)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Persist the USER-CONFIRMED mapping (never a suggestion) so a later
    # training job (api/training.py) knows which column is the id/target/
    # revenue/features without re-deriving it - saved regardless of
    # can_proceed so re-validating after a fix always reflects the latest
    # confirmed mapping.
    row.column_mapping_json = request.column_mapping
    db.commit()

    advance_status(db, current_user.tenant_id, "data_validated", upload_id=row.id)

    return {"upload_id": row.id, "filename": row.filename, **result}


@router.post(
    "/confirm",
    summary="Confirm the validation review and finalize onboarding for now",
    description="Marks your company's onboarding as `ready` - called when the user clicks "
    "'Continue' (or 'Proceed Anyway' after acknowledging a data-sufficiency warning) on the "
    "validation report. A no-op for tenants that never registered via "
    "POST /api/companies/register. Does NOT train a model - see this module's top-level "
    "docstring for Stage 1's scope.",
)
def confirm_onboarding(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    row = advance_status(db, current_user.tenant_id, "ready")
    if row is None:
        raise HTTPException(status_code=404, detail="No onboarding in progress for this tenant.")
    return {"tenant_id": row.tenant_id, "step": row.step, "updated_at": row.updated_at.isoformat()}
