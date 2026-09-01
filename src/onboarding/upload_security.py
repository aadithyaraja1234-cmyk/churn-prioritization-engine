"""Upload security checks for self-service CSV onboarding (Stage 1).

SCOPE - what this defends against, and what it explicitly does NOT:
  - Defends against: oversized uploads, non-CSV/binary content disguised
    with a .csv extension or a spoofed Content-Type, null-byte/non-text
    payloads, and CSV/Excel formula-injection payloads (a cell value that
    would execute as a formula if this data is later opened in Excel/
    Google Sheets, e.g. "=cmd|'/c calc'!A1").
  - Does NOT defend against: malware embedded in file bytes that isn't a
    formula-injection or binary-content signal (e.g. a polyglot file that
    still decodes as plausible CSV text). That requires a real content
    scanning service (e.g. ClamAV) integrated as a separate, later addition
    - it is explicitly out of scope for Stage 1 and must never be implied
    to already exist by this module's docstring, error messages, or tests.

FORMULA-INJECTION POLICY: reject-and-explain, not silently sanitize. A
newly onboarding, security-conscious customer is better served by an
honest "this cell looks like a spreadsheet formula, please fix it and
re-upload" than by us silently rewriting their data (which risks a user
believing their upload succeeded byte-for-byte when it didn't, and which
this project's own honesty discipline - see other modules' docstrings -
argues against). Sanitization is friendlier but hides a real problem in
the source data; for a first version we choose the safer, more honest
option.

The size cap is enforced via a genuine streamed chunk-by-chunk read with a
hard byte ceiling, so an oversized file is never assembled in memory -
enforcement stops the instant the cumulative byte count crosses the limit,
regardless of what any Content-Length header claims.
"""

from __future__ import annotations

import csv
import io

from fastapi import UploadFile

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_ROWS = 50_000
READ_CHUNK_SIZE = 1024 * 1024  # 1MB per chunk while streaming

# A cell value starting with any of these is a spreadsheet-formula trigger
# in Excel/Google/LibreOffice - the classic CSV-injection vector (e.g.
# "=cmd|'/c calc'!A1", "+1+1", "-2+3", "@SUM(1,9)").
FORMULA_INJECTION_PREFIXES = ("=", "+", "-", "@")

# A genuine CSV/text file should be almost entirely printable ASCII/UTF-8
# text plus normal whitespace. A lower ratio is a strong signal of binary
# content disguised behind a .csv extension or a spoofed content-type -
# neither of which this module trusts.
MIN_PRINTABLE_RATIO = 0.95


class UploadRejected(Exception):
    """Raised with a specific, user-facing reason - callers should map this
    directly to an honest 400 response, never a generic 500."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


async def read_and_validate_upload(file: UploadFile) -> str:
    """Streams `file` in fixed-size chunks (never materializing more than
    MAX_FILE_SIZE_BYTES in memory), then validates it is genuine CSV text
    free of null bytes, binary content, and formula-injection payloads.
    Returns the raw CSV text on success; raises UploadRejected otherwise."""
    total_bytes = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(READ_CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_FILE_SIZE_BYTES:
            raise UploadRejected(
                f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB upload limit."
            )
        if b"\x00" in chunk:
            raise UploadRejected("File contains null bytes - this is not valid CSV text.")
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    if not raw_bytes.strip():
        raise UploadRejected("File is empty.")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise UploadRejected(
            "File is not valid UTF-8 text - this looks like a binary file, not a CSV. "
            "(Note: this checks for malformed/binary content, not malware - a real "
            "antivirus/content-scanning service would be a separate, later addition.)"
        )

    printable_count = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
    if printable_count / max(len(text), 1) < MIN_PRINTABLE_RATIO:
        raise UploadRejected("File content does not look like plain-text CSV data.")

    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise UploadRejected(f"File could not be parsed as CSV: {exc}")

    if len(rows) < 2:
        raise UploadRejected("File must contain a header row and at least one data row.")
    if len(rows) - 1 > MAX_ROWS:
        raise UploadRejected(f"File has more than the {MAX_ROWS:,}-row limit.")

    header = rows[0]
    for row_number, row in enumerate(rows[1:], start=1):
        for column_index, cell in enumerate(row):
            stripped = cell.strip()
            if stripped and stripped[0] in FORMULA_INJECTION_PREFIXES:
                column_name = header[column_index] if column_index < len(header) else f"column {column_index + 1}"
                raise UploadRejected(
                    f"Row {row_number}, column '{column_name}' contains a value starting with "
                    f"'{stripped[0]}', which looks like a spreadsheet formula. We don't allow "
                    "formula-like values for security reasons (formula injection risk if this "
                    "data is later opened in Excel/Sheets) - please remove or re-quote this "
                    "value and re-upload."
                )

    return text
