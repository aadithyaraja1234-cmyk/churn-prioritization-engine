"""Field-level validation for company self-registration
(POST /api/companies/register). Returns a specific per-field error message
(or None) rather than raising - the caller collects every field's result
into one {"errors": {...}} response, so a user sees every problem at once
instead of one generic 400 and a guessing game.

Password policy: minimum length + character-class diversity, not length
alone - a common, defensible baseline (comparable to NIST SP 800-63B's
length-first guidance plus a class-diversity floor) chosen for a first
version. This is intentionally mirrored (not shared via import - the
frontend has no Python runtime) in
frontend/src/utils/passwordStrength.js for the real-time strength meter;
if this policy changes, that file must change with it.
"""

from __future__ import annotations

import re

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

MIN_PASSWORD_LENGTH = 10
MIN_PASSWORD_CHARACTER_CLASSES = 3  # of: lowercase, uppercase, digit, symbol

MIN_COMPANY_NAME_LENGTH = 2
MAX_COMPANY_NAME_LENGTH = 100


def validate_email_format(email: str) -> str | None:
    if not email or not EMAIL_PATTERN.match(email.strip()):
        return "Enter a valid email address."
    return None


def _password_character_classes(password: str) -> int:
    return sum(
        [
            any(c.islower() for c in password),
            any(c.isupper() for c in password),
            any(c.isdigit() for c in password),
            any(not c.isalnum() for c in password),
        ]
    )


def validate_password_strength(password: str) -> str | None:
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if _password_character_classes(password) < MIN_PASSWORD_CHARACTER_CLASSES:
        return (
            f"Password must include at least {MIN_PASSWORD_CHARACTER_CLASSES} of: "
            "lowercase letters, uppercase letters, numbers, symbols."
        )
    return None


def validate_company_name(name: str) -> str | None:
    stripped = (name or "").strip()
    if len(stripped) < MIN_COMPANY_NAME_LENGTH:
        return f"Company name must be at least {MIN_COMPANY_NAME_LENGTH} characters."
    if len(stripped) > MAX_COMPANY_NAME_LENGTH:
        return f"Company name must be under {MAX_COMPANY_NAME_LENGTH} characters."
    return None


def slugify_company_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "company"
