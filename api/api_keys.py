"""API-key auth: a second, independent auth method alongside JWT (api/auth.py).

Account management (create/list/revoke) is JWT-only - a human manages their
own tenant's keys, same as any other authenticated action. The generated key
itself is a separate credential that a system integration then uses on its
own, without ever logging in.

Only a SHA-256 hash of each raw key is ever stored, never the key itself. A
fast hash (not bcrypt) is deliberate here: unlike a human-chosen password, an
API key is generated with 32 bytes of real randomness (secrets.token_urlsafe),
so it already has enough entropy that a slow, salted hash buys nothing - and
a fast hash is what makes an indexed key_hash lookup per-request cheap.

get_current_user_or_api_key() is the auth-method-agnostic dependency: it
tries a JWT bearer token first, falls back to an X-API-Key header, and
returns the same CurrentUser shape either way, so downstream endpoint logic
never needs to know or care which auth method was used.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import CurrentUser, bearer_scheme, get_current_user
from database.db import get_db, get_tenant_scoped_query
from database.models import ApiKey

API_KEY_PREFIX = "sk_live_"
DEFAULT_RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_WINDOW_SECONDS = 60

router = APIRouter(prefix="/api/api-keys", tags=["API Keys"])


class ApiKeyCreateRequest(BaseModel):
    name: str
    rate_limit_per_minute: int | None = None


class ApiKeyCreateResponse(BaseModel):
    id: int
    name: str
    api_key: str
    masked_key: str
    rate_limit_per_minute: int
    created_at: str


class ApiKeyListItem(BaseModel):
    id: int
    name: str
    masked_key: str
    rate_limit_per_minute: int
    created_at: str
    last_used_at: str | None
    revoked: bool
    revoked_at: str | None


class ApiKeyRevokeResponse(BaseModel):
    id: int
    revoked_at: str


def _generate_raw_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _mask_key(raw_key: str) -> str:
    return f"{API_KEY_PREFIX}....{raw_key[-4:]}"


@router.post(
    "",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key for your tenant",
    description="Generates a new API key scoped to your own tenant and returns the RAW key "
    "in this response ONLY - it is never retrievable again afterward (only its hash is "
    "stored). Copy it somewhere safe immediately. JWT-authenticated only - this is account "
    "management, not something an API key can do to itself.",
)
def create_api_key(
    request: ApiKeyCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiKeyCreateResponse:
    raw_key = _generate_raw_key()
    rate_limit = request.rate_limit_per_minute or DEFAULT_RATE_LIMIT_PER_MINUTE

    row = ApiKey(
        tenant_id=current_user.tenant_id,
        name=request.name,
        key_hash=_hash_key(raw_key),
        masked_key=_mask_key(raw_key),
        rate_limit_per_minute=rate_limit,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return ApiKeyCreateResponse(
        id=row.id,
        name=row.name,
        api_key=raw_key,
        masked_key=row.masked_key,
        rate_limit_per_minute=row.rate_limit_per_minute,
        created_at=row.created_at.isoformat(),
    )


@router.get(
    "",
    response_model=list[ApiKeyListItem],
    summary="List your tenant's API keys",
    description="Returns every API key (active and revoked) created for your tenant - name, "
    "creation/last-used timestamps, and a masked value (e.g. `sk_live_....a3f2`). Never "
    "returns the raw key; that is only ever shown once, at creation time.",
)
def list_api_keys(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[ApiKeyListItem]:
    rows = (
        get_tenant_scoped_query(db, ApiKey, tenant_id=current_user.tenant_id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return [
        ApiKeyListItem(
            id=row.id,
            name=row.name,
            masked_key=row.masked_key,
            rate_limit_per_minute=row.rate_limit_per_minute,
            created_at=row.created_at.isoformat(),
            last_used_at=row.last_used_at.isoformat() if row.last_used_at else None,
            revoked=row.revoked_at is not None,
            revoked_at=row.revoked_at.isoformat() if row.revoked_at else None,
        )
        for row in rows
    ]


@router.delete(
    "/{key_id}",
    response_model=ApiKeyRevokeResponse,
    summary="Revoke an API key",
    description="Soft-deletes an API key by setting its revoked_at timestamp - the row is "
    "kept for audit purposes, never hard-deleted. A revoked key is rejected immediately on "
    "every subsequent request. Returns 404 if no key with that id exists for your tenant.",
)
def revoke_api_key(
    key_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiKeyRevokeResponse:
    row = (
        get_tenant_scoped_query(db, ApiKey, tenant_id=current_user.tenant_id)
        .filter(ApiKey.id == key_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")

    if row.revoked_at is None:
        row.revoked_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)

    return ApiKeyRevokeResponse(id=row.id, revoked_at=row.revoked_at.isoformat())


# In-memory sliding-window rate limiter, keyed by api_keys.id. A simple dict
# is fine at this scale (single-process dev/demo deployment) - see Part 2
# item 5 of the feature spec this implements. Not persisted, not shared
# across processes; a restart or multi-worker deployment would reset/split
# the window, which is an accepted limitation at this scale.
_RATE_LIMIT_WINDOWS: dict[int, list[float]] = {}


def _enforce_rate_limit(api_key_id: int, limit_per_minute: int) -> None:
    now = time.monotonic()
    window = _RATE_LIMIT_WINDOWS.setdefault(api_key_id, [])
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while window and window[0] < cutoff:
        window.pop(0)

    if len(window) >= limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: this API key allows {limit_per_minute} requests per minute",
        )
    window.append(now)


def reset_rate_limits_for_testing() -> None:
    _RATE_LIMIT_WINDOWS.clear()


def _authenticate_api_key(raw_key: str, db: Session) -> CurrentUser:
    row = db.query(ApiKey).filter(ApiKey.key_hash == _hash_key(raw_key)).first()
    if row is None or row.revoked_at is not None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    _enforce_rate_limit(row.id, row.rate_limit_per_minute)

    row.last_used_at = datetime.now(timezone.utc)
    db.commit()

    # role="api_key" and a synthetic email give downstream code the exact
    # same CurrentUser shape a JWT would - no endpoint logic branches on
    # which auth method was used, only on tenant_id.
    return CurrentUser(email=f"api-key:{row.name}", tenant_id=row.tenant_id, role="api_key")


def get_current_user_or_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> CurrentUser:
    """Accept EITHER a JWT bearer token OR an X-API-Key header, returning the
    same CurrentUser shape either way. Only used on the deliberate subset of
    endpoints that support API-key access - everything else (auth, Copilot,
    API-key management itself) depends on get_current_user() directly, which
    only ever looks at the JWT and so rejects an API key even if one is
    supplied alongside it."""
    if credentials is not None:
        return get_current_user(credentials)
    if x_api_key:
        return _authenticate_api_key(x_api_key, db)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated: provide an Authorization: Bearer token or an X-API-Key header",
        headers={"WWW-Authenticate": "Bearer"},
    )
