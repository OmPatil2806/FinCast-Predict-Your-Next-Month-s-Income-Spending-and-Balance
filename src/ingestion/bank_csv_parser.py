"""Parser for a user's own bank CSV export.

Turns any common bank export into the standard schema used across FinCast:
    date (datetime64), description (str), amount (float; expenses negative, income positive)

Handles different separators (; , tab |), date formats (31.01.2026, 2026-01-31, 31/01/2026),
German and English number styles (-1.234,56 and -1,234.56), common English and German
column names, a few preamble lines above the header, and UTF-8 or Windows (cp1252) encodings.
Every problem is raised as ParseError with a message that can be shown to the user.
"""

import csv
import io
import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd

STANDARD_COLUMNS = ["date", "description", "amount"]
DELIMITERS = [";", ",", "\t", "|"]
DATE_FORMATS = ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%y"]
MAX_HEADER_SEARCH_LINES = 30

# Known column names, most preferred first. Matching ignores case and surrounding spaces.
COLUMN_ALIASES = {
    "date": [
        "date", "booking date", "transaction date", "posting date",
        "buchungstag", "buchungsdatum", "datum", "value date", "valuta", "wertstellung",
    ],
    "description": [
        "description", "details", "merchant", "payee", "name",
        "beguenstigter/zahlungspflichtiger", "begünstigter/zahlungspflichtiger",
        "empfänger", "empfaenger", "auftraggeber", "verwendungszweck",
        "buchungstext", "memo", "reference", "text",
    ],
    "amount": ["amount", "betrag", "betrag (eur)", "amount (eur)", "umsatz", "value"],
}


class ParseError(ValueError):
    """Raised with a user-friendly message when a bank file cannot be read."""


# ---------- reading the raw file ----------

def _read_text(source: str | Path | bytes | BinaryIO) -> str:
    """Return the file content as text, trying UTF-8 first and then Windows encoding."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:  # file-like object, e.g. Streamlit's UploadedFile
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
    if not raw or not raw.strip():
        raise ParseError("The file is empty.")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError("Could not read the file. Please export it as CSV (UTF-8).")


def detect_delimiter(text: str) -> str:
    """Guess the column separator from the first lines of the file."""
    sample = "\n".join(text.splitlines()[:MAX_HEADER_SEARCH_LINES])
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        # fall back to the separator that appears most often
        counts = {d: sample.count(d) for d in DELIMITERS}
        best = max(counts, key=counts.get)
        if counts[best] == 0:
            raise ParseError("Could not detect the column separator. Is this a CSV file?")
        return best


def _normalise(name: str) -> str:
    return str(name).strip().strip('"').lower()


def guess_columns(headers: list[str]) -> dict[str, str]:
    """Map 'date', 'description' and 'amount' to the file's own column names where recognised."""
    normalised = {_normalise(h): h for h in headers}
    mapping = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised and normalised[alias] not in mapping.values():
                mapping[target] = normalised[alias]
                break
    return mapping


def _find_header_line(lines: list[str], delimiter: str) -> int:
    """Index of the header row: the first line that names a date and an amount column."""
    for i, line in enumerate(lines[:MAX_HEADER_SEARCH_LINES]):
        cells = next(csv.reader([line], delimiter=delimiter), [])
        found = guess_columns(cells)
        if "date" in found and "amount" in found:
            return i
    return 0


def read_raw_csv(source: str | Path | bytes | BinaryIO) -> tuple[pd.DataFrame, int]:
    """Read the file with every cell as text (no type guessing).

    Returns the table and the number of lines skipped above the header.
    Used by the Import page for the preview and the column-mapping dropdowns.
    """
    text = _read_text(source)
    delimiter = detect_delimiter(text)
    lines = text.splitlines()
    header_line = _find_header_line(lines, delimiter)
    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines[header_line:])),
            sep=delimiter,
            dtype=str,              # keep "01.07.2026" and "3.200,00" exactly as written
            keep_default_na=False,
            skip_blank_lines=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as err:
        raise ParseError(f"Could not read the table: {err}") from err
    df.columns = [str(c).strip() for c in df.columns]
    return df, header_line


# ---------- parsing values ----------

_NOT_NUMBER = re.compile(r"[^0-9,.\-+]")


def parse_amount(value: str) -> float:
    """Convert a money string in German or English style to a float.

    '-1.234,56' -> -1234.56, '-1,234.56' -> -1234.56, '42,37' -> 42.37,
    '€ -12,00' -> -12.0, '12,00-' -> -12.0, '(12.00)' -> -12.0
    """
    text = str(value).strip()
    if not text:
        raise ValueError("empty amount")
    negative = (text.startswith("(") and text.endswith(")")) or text.endswith("-")
    cleaned = _NOT_NUMBER.sub("", text).rstrip("-")
    if cleaned.startswith("-"):
        negative = True
    cleaned = cleaned.lstrip("+-")
    if not cleaned or not any(ch.isdigit() for ch in cleaned):
        raise ValueError(f"not a number: {value!r}")

    last_comma, last_dot = cleaned.rfind(","), cleaned.rfind(".")
    if last_comma >= 0 and last_dot >= 0:
        # both present: whichever comes last is the decimal separator
        decimal = "," if last_comma > last_dot else "."
    elif last_comma >= 0 or last_dot >= 0:
        sep = "," if last_comma >= 0 else "."
        digits_after = len(cleaned) - cleaned.rfind(sep) - 1
        # one separator followed by exactly 3 digits ("1.234", "1,234") is a thousands separator
        is_thousands = cleaned.count(sep) > 1 or (digits_after == 3 and not cleaned.startswith("0" + sep))
        decimal = None if is_thousands else sep
    else:
        decimal = None

    thousands = {",": ".", ".": ","}.get(decimal) if decimal else None
    if decimal is None:
        number = cleaned.replace(",", "").replace(".", "")
    else:
        number = cleaned.replace(thousands, "").replace(decimal, ".")
    try:
        result = float(number)
    except ValueError:
        raise ValueError(f"not a number: {value!r}") from None
    return -result if negative else result


class BadDateError(ValueError):
    """A date value that no known format can read; .index is its row label."""

    def __init__(self, value: str, index):
        super().__init__(f"'{value}' is not a valid date")
        self.index = index


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse a column of date strings, trying common European and ISO formats.

    Raises BadDateError for the first value that the best-matching format cannot read.
    """
    stripped = values.astype(str).str.strip()
    attempts = [pd.to_datetime(stripped, format=fmt, errors="coerce") for fmt in DATE_FORMATS]
    best = max(attempts, key=lambda parsed: parsed.notna().sum())
    if best.notna().all():
        return best
    first_bad = best.isna().idxmax()
    raise BadDateError(stripped[first_bad], first_bad)


# ---------- main entry point ----------

def parse_bank_csv(source: str | Path | bytes | BinaryIO,
                   column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a bank CSV and return a DataFrame with columns date, description, amount.

    column_map lets the user choose the columns, e.g.
    {"date": "Buchungstag", "description": "Verwendungszweck", "amount": "Betrag"}.
    Without it, the columns are guessed from their names.
    """
    df, skipped = read_raw_csv(source)
    mapping = column_map or guess_columns(list(df.columns))

    for target in STANDARD_COLUMNS:
        if target not in mapping:
            article = "an" if target == "amount" else "a"
            raise ParseError(
                f"Could not find {article} {target} column. Columns in the file: {', '.join(df.columns)}. "
                "Please choose the columns manually."
            )
        if mapping[target] not in df.columns:
            raise ParseError(f"Column '{mapping[target]}' does not exist in the file.")

    out = pd.DataFrame({target: df[mapping[target]].astype(str).str.strip() for target in STANDARD_COLUMNS})
    out = out[(out != "").any(axis=1)]  # ignore rows where all three fields are empty
    if out.empty:
        raise ParseError("The file has no transactions.")

    try:
        out["date"] = parse_dates(out["date"])
    except BadDateError as err:
        raise ParseError(f"Row {err.index + skipped + 2}: {err}.") from None

    # row numbers as the user sees them in a spreadsheet: header is line 1 after any skipped lines
    amounts = []
    for idx, raw in out["amount"].items():
        try:
            amounts.append(parse_amount(raw))
        except ValueError:
            raise ParseError(f"Row {idx + skipped + 2}: '{raw}' is not a valid amount.") from None
    out["amount"] = amounts

    return out.reset_index(drop=True)
