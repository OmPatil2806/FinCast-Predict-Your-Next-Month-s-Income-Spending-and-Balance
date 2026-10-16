"""'Download all my data' (step 8.8): every table of the user as CSV files in one ZIP."""

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app.database.crud import export_user_data

README = """FinCast data export
Created: {created} (UTC)

One CSV file per table, holding every row stored for your account:
  account.csv          username, email, sign-up date (your password is never exported)
  profiles.csv         income type, currency, opening balance
  transactions.csv     every imported or added transaction, with category and anomaly status
  monthly_summary.csv  monthly totals (if any were stored)
  forecasts.csv        the forecasts FinCast made for you
  goals.csv            your savings goals
"""


def user_data_zip(user_id: int, db_path: str | Path | None = None) -> bytes:
    data = export_user_data(user_id, db_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", README.format(created=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}"))
        for table, frame in data.items():
            archive.writestr(f"{table}.csv", frame.to_csv(index=False))
    return buffer.getvalue()
