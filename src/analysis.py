"""
analysis.py  -  inbox analytics with Pandas groupby / pivot_table, Counter and NumPy.
"""
import numpy as np
import pandas as pd

from src.features import keyword_counter


def domain_summary(df, top=10):
    """Emails per sender domain and % predicted as spam."""
    g = df.groupby("sender_domain").agg(
        emails=("sender", "count"),
        spam_emails=("predicted", lambda s: (s == "spam").sum()),
    )
    g["spam_pct"] = (g["spam_emails"] / g["emails"] * 100).round(1)
    return g.sort_values("emails", ascending=False).head(top)


def spam_by_hour(df):
    """Count of predicted-spam emails per hour of day (0-23)."""
    s = df[df["predicted"] == "spam"].groupby("hour").size()
    return s.reindex(range(24), fill_value=0)


def label_by_day(df):
    """Pivot table: day of week x predicted label."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pt = pd.pivot_table(df, index="day", columns="predicted", values="sender",
                        aggfunc="count", fill_value=0)
    return pt.reindex(order)


def label_by_hour_block(df):
    """Pivot table: 6-hour block of day x predicted label."""
    bins = [-1, 5, 11, 17, 23]
    names = ["Night (0-5)", "Morning (6-11)", "Afternoon (12-17)", "Evening (18-23)"]
    block = pd.cut(df["hour"], bins=bins, labels=names)
    return pd.crosstab(block, df["predicted"])


def top_spam_keywords(df, n=15):
    """Top keywords (Counter) in e-mails predicted as spam."""
    texts = (df.loc[df["predicted"] == "spam", "subject"] + " " +
             df.loc[df["predicted"] == "spam", "body"]).tolist()
    return keyword_counter(texts).most_common(n)


def score_stats(scores):
    """NumPy statistics of the spam scores."""
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "max": float(np.max(scores)),
    }
