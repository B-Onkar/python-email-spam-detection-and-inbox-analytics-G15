"""
cleaning.py  -  load and clean the raw email CSV with Pandas + regex.
"""
import re
from datetime import datetime

import pandas as pd

# ---- Regex patterns (compiled once) ---------------------------------
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")          # valid sender
DATE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}-[A-Za-z]{3}-\d{2})"
                     r"(?:\s+(\d{1,2}):(\d{2}))?\s*$")            # 3 date styles + optional HH:MM
WHITESPACE_RE = re.compile(r"\s+")

# date style -> strptime format
DATE_FORMATS = {
    "iso": "%Y-%m-%d",
    "slash": "%d/%m/%Y",
    "dmy_mon": "%d-%b-%y",
}


def is_valid_email(address: str) -> bool:
    """True if the sender looks like a valid e-mail address."""
    return bool(EMAIL_RE.match(address or ""))


def parse_date(text):
    """Convert 2025-03-01 / 01/03/2025 / 1-Mar-25 (+ optional HH:MM) to datetime.
    Returns pandas.NaT if the text cannot be parsed."""
    if not isinstance(text, str):
        return pd.NaT
    m = DATE_RE.match(text)
    if not m:
        return pd.NaT
    date_part, hh, mm = m.group(1), m.group(2), m.group(3)
    if re.match(r"\d{4}-", date_part):
        fmt = DATE_FORMATS["iso"]
    elif "/" in date_part:
        fmt = DATE_FORMATS["slash"]
    else:
        fmt = DATE_FORMATS["dmy_mon"]
    try:
        dt = datetime.strptime(date_part, fmt)
    except ValueError:
        return pd.NaT
    if hh is not None:
        dt = dt.replace(hour=int(hh), minute=int(mm))
    return pd.Timestamp(dt)


def clean_emails(path: str):
    """Load the raw CSV and return (clean_df, report_dict)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    report = {"raw_rows": len(raw)}

    # 1. strip whitespace / collapse spaces in text columns
    for col in ["sender", "subject", "body", "label"]:
        raw[col] = raw[col].fillna("").map(lambda s: WHITESPACE_RE.sub(" ", s).strip())
    raw["date"] = raw["date"].fillna("").str.strip()

    # 2. drop exact duplicates
    before = len(raw)
    df = raw.drop_duplicates().copy()
    report["duplicates_removed"] = before - len(df)

    # 3. dates -> one datetime format; drop rows with no usable date
    df["date"] = df["date"].map(parse_date)
    report["missing_dates_dropped"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).copy()

    # 4. missing text -> placeholders; label normalised
    report["missing_subject"] = int((df["subject"] == "").sum())
    report["missing_body"] = int((df["body"] == "").sum())
    df["subject"] = df["subject"].replace("", "(no subject)")
    df["label"] = df["label"].str.lower()

    # 5. sender validation flag (kept, not deleted -> invalid sender is a spam signal)
    df["valid_sender"] = df["sender"].map(is_valid_email)
    report["invalid_senders"] = int((~df["valid_sender"]).sum())

    # 6. derived columns
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["sender_domain"] = df.apply(
        lambda r: r["sender"].split("@")[-1].lower() if r["valid_sender"] else "invalid", axis=1)

    df = df.reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report
