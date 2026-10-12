"""Tests for the bank CSV parser: good formats are read correctly, bad files give clear errors."""

import pandas as pd
import pytest

from src.ingestion.bank_csv_parser import (
    ParseError, detect_delimiter, guess_columns, parse_amount, parse_bank_csv,
)
from src.utils.helpers import get_project_root

SAMPLE = get_project_root() / "data" / "sample" / "sample_bank_export.csv"


# ---------- amounts ----------

@pytest.mark.parametrize("text, expected", [
    ("-1.234,56", -1234.56),      # German
    ("-1,234.56", -1234.56),      # English
    ("3.200,00", 3200.0),
    ("42,37", 42.37),
    ("1234.5", 1234.5),
    ("1.234", 1234.0),            # one separator + 3 digits = thousands
    ("1,234", 1234.0),
    ("0,123", 0.123),
    ("€ -12,00", -12.0),
    ("12,00-", -12.0),            # trailing minus
    ("(12.00)", -12.0),           # accounting style
    ("+15,5", 15.5),
    ("1 234,56 EUR", 1234.56),
    ("2.000.000,00", 2_000_000.0),
    ("1,234,567.89", 1_234_567.89),
])
def test_parse_amount(text, expected):
    assert parse_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["abc", "", "EUR", "--"])
def test_parse_amount_rejects_non_numbers(text):
    with pytest.raises(ValueError):
        parse_amount(text)


# ---------- detection ----------

@pytest.mark.parametrize("text, delimiter", [
    ("a;b;c\n1;2;3\n", ";"),
    ("a,b,c\n1,2,3\n", ","),
    ("a\tb\tc\n1\t2\t3\n", "\t"),
    ("a|b|c\n1|2|3\n", "|"),
])
def test_detect_delimiter(text, delimiter):
    assert detect_delimiter(text) == delimiter


def test_guess_columns_english_and_german():
    assert guess_columns(["Date", "Description", "Amount"]) == {
        "date": "Date", "description": "Description", "amount": "Amount"}
    german = ["Buchungstag", "Valuta", "Buchungstext", "Beguenstigter/Zahlungspflichtiger",
              "Verwendungszweck", "Betrag (EUR)"]
    assert guess_columns(german) == {
        "date": "Buchungstag", "description": "Beguenstigter/Zahlungspflichtiger", "amount": "Betrag (EUR)"}


# ---------- good files ----------

def test_sample_file():
    df = parse_bank_csv(SAMPLE)
    assert list(df.columns) == ["date", "description", "amount"]
    assert len(df) == 60
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].min() == pd.Timestamp("2026-07-01")
    assert df["amount"].sum() == pytest.approx(4513.81)
    assert (df.loc[df["description"].str.contains("Gehalt"), "amount"] == 3200.0).all()


def test_english_comma_file_with_quoted_values():
    text = b'Date,Description,Amount\n2026-01-31,Coffee,-3.50\n2026-02-01,"Salary, Jan","2,500.00"\n'
    df = parse_bank_csv(text)
    assert df["description"].tolist() == ["Coffee", "Salary, Jan"]
    assert df["amount"].tolist() == [-3.5, 2500.0]


def test_german_bank_export_with_preamble_and_windows_encoding():
    text = ("Kontoinhaber;Max Mustermann\nZeitraum;01.01.2026 - 31.01.2026\n\n"
            "Buchungstag;Valuta;Buchungstext;Beguenstigter/Zahlungspflichtiger;Verwendungszweck;Betrag (EUR)\n"
            "02.01.2026;02.01.2026;Lastschrift;Bäckerei Müller;Brötchen;-4,20\n").encode("cp1252")
    df = parse_bank_csv(text)
    assert df.iloc[0]["description"] == "Bäckerei Müller"
    assert df.iloc[0]["amount"] == -4.2
    assert df.iloc[0]["date"] == pd.Timestamp("2026-01-02")


def test_blank_rows_are_ignored():
    df = parse_bank_csv(b"Date;Description;Amount\n01.07.2026;REWE;-10,00\n;;\n\n")
    assert len(df) == 1


def test_manual_column_mapping():
    text = b"Tag;Was;Wert\n01.07.2026;REWE;-10,00\n"
    df = parse_bank_csv(text, column_map={"date": "Tag", "description": "Was", "amount": "Wert"})
    assert df.iloc[0]["amount"] == -10.0


def test_accepts_file_like_object(tmp_path):
    path = tmp_path / "export.csv"
    path.write_bytes(SAMPLE.read_bytes())
    with open(path, "rb") as f:
        assert len(parse_bank_csv(f)) == 60


# ---------- broken files: clear message, never a crash ----------

@pytest.mark.parametrize("content, message", [
    (b"", "The file is empty."),
    (b"   \n  \n", "The file is empty."),
    (b"Date;Description;Amount\n", "The file has no transactions."),
    (b"Date;Description;Notes\n01.07.2026;REWE;hello\n", "Could not find an amount column"),
    (b"Day;Description;Amount\n01.07.2026;REWE;-1,00\n", "Could not find a date column"),
    (b"Date;Description;Amount\n01.07.2026;REWE;-10,00\n02.07.2026;LIDL;abc\n",
     "Row 3: 'abc' is not a valid amount."),
    (b"Date;Description;Amount\n01.07.2026;REWE;-10,00\n2026-13-45;LIDL;-5,00\n",
     "Row 3: '2026-13-45' is not a valid date."),
    (b"hello world\nthis is not a table\n", "Could not detect the column separator"),
])
def test_broken_files_give_clear_errors(content, message):
    with pytest.raises(ParseError, match=message.replace("(", r"\(").replace(".", r"\.")):
        parse_bank_csv(content)


def test_mapping_to_missing_column():
    with pytest.raises(ParseError, match="Column 'Datum' does not exist"):
        parse_bank_csv(b"Date;Description;Amount\n01.07.2026;REWE;-10,00\n",
                       column_map={"date": "Datum", "description": "Description", "amount": "Amount"})
