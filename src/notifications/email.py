"""Pluggable transactional email - currently just the password-reset email
(api/auth.py's /forgot-password).

DEGRADE POLICY, same shape as src/copilot/agent.py's GEMINI_API_KEY and
src/onboarding/ai_mapping.py's: unset RESEND_API_KEY doesn't raise or
silently do nothing - it logs the recipient + reset link via
logger.warning so whoever operates this deployment can see it in their
own server logs (e.g. local dev, or before a real email provider is
wired up). The one thing this must NEVER do is return the link in an API
response: /forgot-password's whole point is an out-of-band channel only
the real account owner's inbox receives - putting it in the HTTP response
instead would let any caller enumerate registered emails AND hand them
the reset token directly, defeating the feature entirely.

Resend (https://resend.com) chosen for the real-provider path: a plain
HTTP POST with an API key, no SMTP setup, a genuine free tier (100
emails/day) - the simplest option that still sends a real email, matching
this project's own "free-tier-friendly optional integration" precedent
(GEMINI_API_KEY).
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
# Resend's own shared sandbox sender - works with no domain verification,
# fine for getting this feature actually sending real email immediately;
# switch to a verified sender on your own domain when you have one (see
# .env.example).
RESEND_FROM_ADDRESS = os.environ.get("RESEND_FROM_ADDRESS", "Churn Engine <onboarding@resend.dev>")

_RESEND_TIMEOUT_SECONDS = 10


class EmailSendError(RuntimeError):
    """Raised only when a real provider IS configured but the send call
    itself failed (network error, Resend rejected the request, etc.) -
    never raised just because no provider is configured, since that's the
    log-only fallback above, not an error condition."""


def send_password_reset_email(*, to_email: str, reset_link: str) -> None:
    if not RESEND_API_KEY:
        logger.warning(
            "RESEND_API_KEY not set - password reset email NOT actually sent to %s. "
            "Dev-fallback only: reset link is %s (a real user can't see this - set "
            "RESEND_API_KEY so this becomes a real email; see .env.example).",
            to_email,
            reset_link,
        )
        return

    try:
        response = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": RESEND_FROM_ADDRESS,
                "to": [to_email],
                "subject": "Reset your password",
                "html": (
                    "<p>Someone requested a password reset for this account.</p>"
                    f'<p><a href="{reset_link}">Click here to reset your password</a></p>'
                    "<p>This link expires in 1 hour. If you didn't request this, you can safely "
                    "ignore this email - your password won't be changed.</p>"
                ),
            },
            timeout=_RESEND_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise EmailSendError(f"Resend API call failed: {exc}") from exc
