"""Project-wide pytest fixtures.

reset_rate_limits: api/rate_limit.py's (and api/api_keys.py's) sliding-
window state is a process-global dict, not per-test - the whole test
SUITE shares one process, and many tests each call /auth/register,
/auth/login, and/or /api/companies/register in a real 60-second wall-clock
window (the suite itself runs in minutes, not hours). Without a reset
between every test, login/register's new IP-based rate limit
(api/rate_limit.py, added alongside this fixture) would start rejecting
otherwise-unrelated tests with 429s purely from cumulative call volume
across the whole session - a false failure that has nothing to do with
whatever that individual test is actually checking. autouse=True so every
test gets this for free with no per-file wiring, mirroring the same reset
tests/test_api_keys.py's own `client` fixture already did manually for
api/api_keys.py's rate limiter before this existed.
"""

from __future__ import annotations

import pytest

from api import api_keys as api_keys_module
from api import rate_limit as rate_limit_module


@pytest.fixture(autouse=True)
def reset_rate_limits():
    api_keys_module.reset_rate_limits_for_testing()
    rate_limit_module.reset_rate_limits_for_testing()
    yield
    api_keys_module.reset_rate_limits_for_testing()
    rate_limit_module.reset_rate_limits_for_testing()
