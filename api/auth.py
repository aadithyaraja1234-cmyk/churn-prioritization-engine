"""JWT authentication, enforced centrally via get_current_user().

Every data-returning endpoint must depend on get_current_user() and use its
tenant_id (never a query parameter) when calling get_tenant_scoped_query().
The tenant comes from the verified token, not from user input - see
api/main.py's /customers endpoint for the enforced pattern.

SECRET_KEY falls back to a dev default so this runs out of the box; set the
JWT_SECRET_KEY env var to a real secret before this is ever exposed outside
a local/dev environment.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.rate_limit import enforce_ip_rate_limit
from database.db import get_db
from database.models import User
from src.notifications.email import send_password_reset_email
from src.validation.credentials import validate_password_strength

# Deliberately tighter than api_keys.py's DEFAULT_RATE_LIMIT_PER_MINUTE=60
# (that's for an already-issued, high-entropy API key; this is guarding a
# human-guessable password) - low enough to make a brute-force loop
# impractical, high enough that a real user mistyping their password a few
# times in a row never gets blocked.
LOGIN_RATE_LIMIT_PER_MINUTE = 10
REGISTER_RATE_LIMIT_PER_MINUTE = 5
# Looser than LOGIN (no credential to guess - just "does this token match"),
# tight enough that guessing a 32-byte token by brute force stays wildly
# impractical (2^256 possibilities; a few hundred guesses/minute changes
# nothing).
FORGOT_PASSWORD_RATE_LIMIT_PER_MINUTE = 5
RESET_PASSWORD_RATE_LIMIT_PER_MINUTE = 10
RESET_TOKEN_EXPIRE_MINUTES = 60

logger = logging.getLogger(__name__)

_DEV_SECRET_KEY = "dev-secret-key-do-not-use-in-production"
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", _DEV_SECRET_KEY)
if SECRET_KEY == _DEV_SECRET_KEY:
    # Loud, not silent: every JWT this process issues is forgeable by anyone
    # who reads this source file unless JWT_SECRET_KEY is set for real before
    # this is ever exposed outside a local/dev environment.
    logger.warning(
        "JWT_SECRET_KEY is not set - using the publicly-known dev default. "
        "Every token issued by this process can be forged. Set JWT_SECRET_KEY "
        "to a real secret before deploying this anywhere reachable outside localhost."
    )
ALGORITHM = "HS256"
# 7 days, not the previous 60 minutes - the frontend now persists this token
# in localStorage across page refreshes/browser restarts (see
# frontend/src/auth/AuthContext.jsx's own comment on that tradeoff); a
# 60-minute expiry would have made that persistence nearly pointless (still
# logged out an hour later regardless of the refresh). Overridable via
# ACCESS_TOKEN_EXPIRE_MINUTES for anyone who wants the old, shorter-lived
# behavior back.
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# auto_error=False so a missing Authorization header reaches get_current_user
# as None and can be turned into a 401 - FastAPI's default HTTPBearer raises
# a 403 on a missing header, which is not what "unauthenticated -> 401" means.
bearer_scheme = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    tenant_id: str
    role: str


class RegisterResponse(BaseModel):
    email: str
    tenant_id: str
    role: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUser(BaseModel):
    email: str
    tenant_id: str
    role: str


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_access_token(*, email: str, tenant_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "tenant_id": tenant_id, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Creates a new user for a given tenant, with a password (hashed with bcrypt "
    "before storage) and a role. For onboarding a new interactive user, not for system "
    "integrations (those authenticate with an API key instead - see `/api/api-keys`). Returns "
    "400 if the email is already registered.",
)
def register(http_request: Request, request: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    enforce_ip_rate_limit(http_request, scope="register", limit_per_minute=REGISTER_RATE_LIMIT_PER_MINUTE)

    existing = db.query(User).filter(User.email == request.email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        tenant_id=request.tenant_id,
        role=request.role,
    )
    db.add(user)
    db.commit()

    return RegisterResponse(email=user.email, tenant_id=user.tenant_id, role=user.role)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and obtain a JWT access token",
    description="Verifies email/password and returns a bearer JWT (valid for "
    f"{ACCESS_TOKEN_EXPIRE_MINUTES} minutes) carrying the user's tenant and role. Pass this "
    "token as `Authorization: Bearer <token>` on every subsequent request. For interactive/"
    "human login; system integrations should use an API key instead once one has been created "
    "via `POST /api/api-keys` (which itself still requires this JWT to call). Returns 401 on "
    f"incorrect credentials, 429 if this IP has made {LOGIN_RATE_LIMIT_PER_MINUTE}+ attempts in "
    "the last minute.",
)
def login(http_request: Request, request: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    enforce_ip_rate_limit(http_request, scope="login", limit_per_minute=LOGIN_RATE_LIMIT_PER_MINUTE)

    user = db.query(User).filter(User.email == request.email).first()
    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    token = create_access_token(email=user.email, tenant_id=user.tenant_id, role=user.role)
    return TokenResponse(access_token=token)


class ForgotPasswordRequest(BaseModel):
    email: str


class ForgotPasswordResponse(BaseModel):
    message: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


# Identical wording regardless of whether the email is actually registered -
# this endpoint must never let a caller learn whether a given email has an
# account here (account-enumeration defense), so the happy path and the
# "no such user" path return the exact same response.
_FORGOT_PASSWORD_GENERIC_MESSAGE = "If that email is registered, we've sent a password reset link to it."


def _hash_reset_token(raw_token: str) -> str:
    # Fast hash, not bcrypt - same reasoning as api/api_keys.py's key_hash:
    # secrets.token_urlsafe(32) already has 256 bits of real randomness, so
    # a slow salted hash buys nothing a fast one doesn't already give here.
    return hashlib.sha256(raw_token.encode()).hexdigest()


@router.post(
    "/forgot-password",
    response_model=ForgotPasswordResponse,
    summary="Request a password reset link",
    description="If `email` belongs to a real account, generates a single-use reset token "
    f"(valid {RESET_TOKEN_EXPIRE_MINUTES} minutes) and emails a reset link to it - see "
    "src/notifications/email.py for how that email actually gets sent (and what happens if "
    "RESEND_API_KEY isn't configured). Always returns the same generic 200 message regardless "
    "of whether the email exists, to avoid leaking which emails are registered.",
)
def forgot_password(
    http_request: Request, request: ForgotPasswordRequest, db: Session = Depends(get_db)
) -> ForgotPasswordResponse:
    enforce_ip_rate_limit(http_request, scope="forgot_password", limit_per_minute=FORGOT_PASSWORD_RATE_LIMIT_PER_MINUTE)

    user = db.query(User).filter(User.email == request.email).first()
    if user is not None:
        raw_token = secrets.token_urlsafe(32)
        user.reset_token_hash = _hash_reset_token(raw_token)
        # Naive UTC, matching the column's own comment in database/models.py
        # (SQLite drops tzinfo on round-trip; comparing an aware "now"
        # against a naive value read back from the DB would raise).
        user.reset_token_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            minutes=RESET_TOKEN_EXPIRE_MINUTES
        )
        db.commit()

        frontend_base_url = os.environ.get("FRONTEND_BASE_URL", "http://localhost:5173").rstrip("/")
        reset_link = f"{frontend_base_url}/reset-password?token={raw_token}"
        send_password_reset_email(to_email=user.email, reset_link=reset_link)

    return ForgotPasswordResponse(message=_FORGOT_PASSWORD_GENERIC_MESSAGE)


@router.post(
    "/reset-password",
    response_model=TokenResponse,
    summary="Complete a password reset and log in",
    description="Consumes the single-use token from a `/forgot-password` email, sets a new "
    "password (same strength rules as registration), invalidates the token, and returns a "
    "bearer JWT so the user lands logged in rather than having to log in again right after. "
    "Returns 400 for an invalid, already-used, or expired token.",
)
def reset_password(
    http_request: Request, request: ResetPasswordRequest, db: Session = Depends(get_db)
) -> TokenResponse:
    enforce_ip_rate_limit(http_request, scope="reset_password", limit_per_minute=RESET_PASSWORD_RATE_LIMIT_PER_MINUTE)

    token_hash = _hash_reset_token(request.token)
    user = db.query(User).filter(User.reset_token_hash == token_hash).first()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if user is None or user.reset_token_expires_at is None or user.reset_token_expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This reset link is invalid or has expired - request a new one.",
        )

    password_error = validate_password_strength(request.new_password)
    if password_error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=password_error)

    user.hashed_password = hash_password(request.new_password)
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    db.commit()

    token = create_access_token(email=user.email, tenant_id=user.tenant_id, role=user.role)
    return TokenResponse(access_token=token)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    role = payload.get("role")
    if email is None or tenant_id is None or role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return CurrentUser(email=email, tenant_id=tenant_id, role=role)
