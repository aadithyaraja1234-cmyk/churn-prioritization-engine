"""Part A of self-service data onboarding (Stage 1): secure upload handling.

Covers exactly what src/onboarding/upload_security.py documents it defends
against - oversized files, non-CSV/binary content, null bytes,
formula-injection payloads, and terminal-escape-sequence/control-character
injection - and confirms a genuinely clean CSV (BOM included) passes. Does
NOT test classic file-based malware scanning, since this module explicitly
does not implement that (see its module docstring for why, for a CSV-only
pipeline)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base

CLEAN_CSV = (
    "customerID,tenure,Contract,MonthlyCharges,Churn\n"
    "C-0001,12,Month-to-month,55.50,No\n"
    "C-0002,24,One year,60.10,Yes\n"
    "C-0003,3,Month-to-month,45.00,No\n"
)


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def token(client):
    client.post(
        "/auth/register",
        json={"email": "upload-tester@example.com", "password": "s3cret-pw", "tenant_id": "telco", "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": "upload-tester@example.com", "password": "s3cret-pw"})
    return response.json()["access_token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _upload(client, token, content: bytes, filename="upload.csv", content_type="text/csv"):
    return client.post(
        "/api/onboarding/upload",
        headers=_auth_headers(token),
        files={"file": (filename, content, content_type)},
    )


def test_valid_clean_csv_passes(client, token):
    response = _upload(client, token, CLEAN_CSV.encode("utf-8"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["row_count"] == 3
    assert set(body["columns"]) == {"customerID", "tenure", "Contract", "MonthlyCharges", "Churn"}
    assert "upload_id" in body
    assert len(body["preview_rows"]) == 3
    # Name-similarity suggestions should recognize these as their real roles.
    assert body["suggested_mapping"]["customerID"]["suggested_role"] == "customer_id"
    assert body["suggested_mapping"]["Churn"]["suggested_role"] == "target"
    assert body["suggested_mapping"]["MonthlyCharges"]["suggested_role"] == "revenue"
    # "duration", not "feature" - see schema_fields.py's "tenure" entry
    # comment for why suggesting the more complete role by default (still
    # kept as an ordinary classifier feature either way) replaced the
    # earlier feature-only default.
    assert body["suggested_mapping"]["tenure"]["suggested_role"] == "duration"


def test_oversized_file_rejected_before_full_parse(client, token):
    # 11MB of otherwise-valid-looking CSV rows - comfortably over the 10MB
    # cap, and over the Content-Length pre-check's allowance too, so this
    # should be rejected fast, without ever fully parsing the body.
    row = "C-%07d,1,Month-to-month,50.00,No\n"
    padding_rows = int(11 * 1024 * 1024 / len(row % 0))
    content = ("customerID,tenure,Contract,MonthlyCharges,Churn\n" + "".join(row % i for i in range(padding_rows))).encode(
        "utf-8"
    )
    assert len(content) > 10 * 1024 * 1024

    response = _upload(client, token, content)
    assert response.status_code in (400, 413), response.text
    assert "10MB" in response.text or "limit" in response.text.lower()


def test_row_count_over_limit_rejected(client, token):
    header = "customerID,tenure,Contract,MonthlyCharges,Churn\n"
    rows = "".join(f"C-{i},1,Month-to-month,50.00,No\n" for i in range(50_001))
    content = (header + rows).encode("utf-8")
    assert len(content) < 10 * 1024 * 1024  # well under the size cap - this is purely a row-count rejection

    response = _upload(client, token, content)
    assert response.status_code == 400
    assert "50,000" in response.text or "row" in response.text.lower()


def test_binary_content_rejected_despite_csv_extension_and_content_type(client, token):
    # Genuine binary bytes that are NOT valid UTF-8 text (0x80-0xFF, no null
    # bytes - that's a separate, dedicated check below), disguised with a
    # .csv filename and a text/csv content-type - neither is trusted.
    binary_content = bytes(range(0x80, 0xFF)) * 100

    response = _upload(client, token, binary_content, filename="not_really.csv", content_type="text/csv")
    assert response.status_code == 400
    assert "binary" in response.text.lower() or "utf-8" in response.text.lower()


def test_null_byte_content_rejected(client, token):
    content = b"customerID,note\nC-0001,\x00bad\n"
    response = _upload(client, token, content)
    assert response.status_code == 400
    assert "null byte" in response.text.lower()


@pytest.mark.parametrize(
    "payload",
    [
        "=cmd|'/c calc'!A1",
        "+1+1",
        "-2+3",
        "@SUM(1,9)",
    ],
)
def test_formula_injection_payloads_are_rejected_with_clear_explanation(client, token, payload):
    content = f"customerID,note\nC-0001,{payload}\n".encode("utf-8")
    response = _upload(client, token, content)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"].lower()
    assert "formula" in detail
    assert "security" in detail


def test_formula_injection_in_quoted_cell_is_still_caught(client, token):
    # A quoted CSV cell still starts with the trigger character once parsed -
    # quoting must not be a bypass.
    content = 'customerID,note\nC-0001,"=cmd|\'/c calc\'!A1"\n'.encode("utf-8")
    response = _upload(client, token, content)
    assert response.status_code == 400
    assert "formula" in response.json()["detail"].lower()


# --- Control-character / terminal-escape-sequence injection: MIN_PRINTABLE_
# RATIO alone is an aggregate check that a handful of malicious control
# characters in an otherwise large, normal-looking file can slip past (see
# upload_security.py's CONTROL_CHARACTER_POLICY docstring) - checked
# per-cell instead. ---


def test_ansi_escape_sequence_in_a_small_file_is_rejected(client, token):
    content = "customerID,note\nC-0001,\x1b[2Jnormal-looking otherwise\n".encode("utf-8")
    response = _upload(client, token, content)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"].lower()
    assert "control character" in detail
    assert "security" in detail


def test_ansi_escape_sequence_hidden_in_a_large_otherwise_clean_file_is_still_caught(client, token):
    """The exact false-negative CONTROL_CHARACTER_POLICY documents: a
    single malicious escape sequence buried in a realistically large file
    barely moves the aggregate printable ratio - must still be caught by
    the per-cell check, not silently let through because the file is
    mostly clean."""
    header = "customerID,note\n"
    rows = [f"C-{i:04d},normal text here more text padding padding padding\n" for i in range(2000)]
    rows[1000] = "C-1000,\x1b]0;pwned\x07title-bar injection\n"
    content = (header + "".join(rows)).encode("utf-8")
    response = _upload(client, token, content)
    assert response.status_code == 400, response.text
    assert "control character" in response.json()["detail"].lower()


def test_utf8_bom_is_stripped_not_rejected_as_a_control_character(client, token):
    """A real, common pattern: a CSV exported from Excel on Windows often
    has a leading UTF-8 BOM. Must be transparently stripped, not rejected
    as a control character in the first cell (see upload_security.py's
    "utf-8-sig" decode)."""
    content = b"\xef\xbb\xbf" + CLEAN_CSV.encode("utf-8")
    response = _upload(client, token, content)
    assert response.status_code == 200, response.text
    assert response.json()["columns"][0] == "customerID"


def test_empty_file_rejected(client, token):
    response = _upload(client, token, b"")
    assert response.status_code == 400


def test_header_only_file_rejected(client, token):
    response = _upload(client, token, b"customerID,tenure\n")
    assert response.status_code == 400
    assert "data row" in response.text.lower()


def test_upload_requires_authentication(client):
    response = client.post("/api/onboarding/upload", files={"file": ("upload.csv", CLEAN_CSV.encode(), "text/csv")})
    assert response.status_code == 401
