"""Upload security checks for self-service CSV onboarding (Stage 1).

SCOPE - what this defends against, and what it explicitly does NOT:
  - Defends against: oversized uploads, non-CSV/binary content disguised
    with a .csv extension or a spoofed Content-Type, null-byte/non-text
    payloads, CSV/Excel formula-injection payloads (a cell value that
    would execute as a formula if this data is later opened in Excel/
    Google Sheets, e.g. "=cmd|'/c calc'!A1"), and terminal/ANSI-escape-
    sequence injection (a cell value that would manipulate a terminal if
    this data is later `cat`/`grep`'d or shown in a log viewer that
    doesn't sanitize control characters - see CONTROL_CHARACTER_POLICY
    below).
  - Does NOT defend against: a real classic-malware payload (an
    executable, an Office-macro document, an image with an embedded
    exploit) smuggled inside file bytes that otherwise still decode as
    plausible UTF-8 CSV text. A traditional content-scanning service
    (e.g. ClamAV) is the standard defense for that - deliberately NOT
    added here: this endpoint only ever accepts and stores CSV/plaintext
    (binary content is already hard-rejected below, so the classic
    malware types a scanner like ClamAV looks for structurally cannot
    reach this far), the parsed cell values are only ever read by
    pandas.read_csv() and rendered as plain React text (no
    dangerouslySetInnerHTML anywhere in frontend/src - verified, not
    assumed - so there is no stored-XSS execution path for a cell value
    either), and the actual planned hosting (Render's free tier) can't
    run a ClamAV sidecar process alongside this app regardless. A real
    content scanner remains the right call if this ever accepts
    non-CSV file types - it is explicitly out of scope for a CSV-only
    pipeline and must never be implied to already exist by this module's
    docstring, error messages, or tests.

FORMULA-INJECTION POLICY: reject-and-explain, not silently sanitize. A
newly onboarding, security-conscious customer is better served by an
honest "this cell looks like a spreadsheet formula, please fix it and
re-upload" than by us silently rewriting their data (which risks a user
believing their upload succeeded byte-for-byte when it didn't, and which
this project's own honesty discipline - see other modules' docstrings -
argues against). Sanitization is friendlier but hides a real problem in
the source data; for a first version we choose the safer, more honest
option. The same reject-and-explain policy applies to the control-
character check below, for the same reason.

CONTROL_CHARACTER_POLICY: MIN_PRINTABLE_RATIO below is an AGGREGATE check
over the whole file - a handful of malicious control characters (e.g. a
few ANSI escape sequences: `\\x1b[2J`, terminal title-bar injection via
`\\x1b]0;...\\x07`, etc.) sprinkled into an otherwise large, normal-looking
file barely move that ratio and sail through undetected (verified
directly: a handful of such sequences embedded in a realistic ~450KB/8000-
row file kept the ratio at 0.9999, nowhere near tripping the 0.95 floor).
Checked per-CELL instead, in the same pass as the formula-injection check
below, so a small malicious payload can't hide inside a large clean file.

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

# str.isprintable() is False for \t/\r/\n too (only the ASCII space counts
# as printable whitespace per Python's own definition) - these three are
# common and legitimate inside a quoted CSV cell, so they're allow-listed
# here exactly like the aggregate MIN_PRINTABLE_RATIO check above already
# does. Every OTHER non-printable character (ANSI/terminal escape
# sequences, zero-width/format characters, etc.) is rejected per-cell -
# see CONTROL_CHARACTER_POLICY above for why the aggregate ratio check
# alone isn't enough to catch a small payload in a large file.
_ALLOWED_CONTROL_WHITESPACE = "\t\r\n"


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
        # "utf-8-sig" rather than plain "utf-8": transparently strips a
        # leading UTF-8 BOM if present (behaves identically to "utf-8"
        # otherwise) - a real, common pattern for a CSV exported from
        # Excel on Windows. Without this, that legitimate BOM byte
        # sequence decodes to a literal U+FEFF character prepended to the
        # first cell, which the new per-cell control-character check below
        # would otherwise reject as a false positive.
        text = raw_bytes.decode("utf-8-sig")
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
            column_name = header[column_index] if column_index < len(header) else f"column {column_index + 1}"

            bad_char = next(
                (ch for ch in cell if not ch.isprintable() and ch not in _ALLOWED_CONTROL_WHITESPACE), None
            )
            if bad_char is not None:
                raise UploadRejected(
                    f"Row {row_number}, column '{column_name}' contains a non-printable control "
                    f"character ({bad_char!r}). We don't allow these for security reasons "
                    "(terminal-escape-sequence injection risk if this data is later viewed with "
                    "cat/grep or a log viewer) - please remove it and re-upload."
                )

            stripped = cell.strip()
            if stripped and stripped[0] in FORMULA_INJECTION_PREFIXES:
                raise UploadRejected(
                    f"Row {row_number}, column '{column_name}' contains a value starting with "
                    f"'{stripped[0]}', which looks like a spreadsheet formula. We don't allow "
                    "formula-like values for security reasons (formula injection risk if this "
                    "data is later opened in Excel/Sheets) - please remove or re-quote this "
                    "value and re-upload."
                )

    return text
