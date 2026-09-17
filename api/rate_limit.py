"""Generic IP-keyed sliding-window rate limiter for unauthenticated
endpoints (login, registration) - there is no api_key_id or user_id yet to
key on at this point in the request, unlike api/api_keys.py's per-key
limiter (which this mirrors: same sliding-window algorithm, same
REDIS_URL-shared-vs-in-memory fallback, same reasoning - kept as a
separate, string-keyed module rather than generalizing api_keys.py's
int-keyed one in place, to avoid touching that already-tested module for
an unrelated caller).

WHY THIS EXISTS: /auth/login has no protection against a brute-force
password-guessing loop without this - an attacker could otherwise try
passwords as fast as the network allows. /auth/register and
/api/companies/register get the same treatment against registration-spam/
account-enumeration abuse.

IP EXTRACTION CAVEAT: keyed on the first hop of X-Forwarded-For if present,
else the raw socket peer address. This is what every reverse-proxied
deployment needs (Render, and most other real hosts, sit the app behind a
proxy that sets X-Forwarded-For to the real client IP - request.client.host
alone would be the proxy's own address, and every request would share one
bucket). It is NOT spoof-proof if the app is ever reachable directly
without a trusted proxy in front of it (a direct caller could set their own
X-Forwarded-For) - acceptable for this app's real deployment shape, but
worth knowing if that ever changes.
"""

from __future__ import annotations

import os
import time

from fastapi import HTTPException, Request

RATE_LIMIT_WINDOW_SECONDS = 60

_WINDOWS: dict[str, list[float]] = {}

REDIS_URL = os.environ.get("REDIS_URL")
_redis_client = None
if REDIS_URL:
    import redis

    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)


def client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_in_memory(key: str, limit_per_minute: int) -> None:
    now = time.monotonic()
    window = _WINDOWS.setdefault(key, [])
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while window and window[0] < cutoff:
        window.pop(0)

    if len(window) >= limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts - try again in a minute (limit: {limit_per_minute}/minute).",
        )
    window.append(now)


def _enforce_redis(key: str, limit_per_minute: int) -> None:
    redis_key = f"ratelimit:ip:{key}"
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    _redis_client.zremrangebyscore(redis_key, 0, cutoff)
    if _redis_client.zcard(redis_key) >= limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts - try again in a minute (limit: {limit_per_minute}/minute).",
        )
    _redis_client.zadd(redis_key, {str(now): now})
    _redis_client.expire(redis_key, RATE_LIMIT_WINDOW_SECONDS)


def enforce_ip_rate_limit(request: Request, *, scope: str, limit_per_minute: int) -> None:
    """Raises 429 if `client_ip(request)` has made >= limit_per_minute
    requests to this `scope` (e.g. "login", "register") in the trailing
    60 seconds; otherwise records this request and returns. `scope` keeps
    a login limit and a register limit from sharing one bucket for the
    same IP."""
    key = f"{scope}:{client_ip(request)}"
    if _redis_client is not None:
        _enforce_redis(key, limit_per_minute)
    else:
        _enforce_in_memory(key, limit_per_minute)


def reset_rate_limits_for_testing() -> None:
    _WINDOWS.clear()
    if _redis_client is not None:
        for key in _redis_client.scan_iter("ratelimit:ip:*"):
            _redis_client.delete(key)
