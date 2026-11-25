"""
main.py  -  runs the whole project:
generate data (if missing) -> clean -> extract features -> score/classify
-> accuracy + stats -> analysis tables -> save charts + cleaned CSV
"""
import os
import subprocess
import sys

import pandas as pd

from src import analysis, cleaning, features, visuals

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "emails.csv")
OUT = os.path.join(BASE, "outputs")

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def banner(text):
    print("\n" + "=" * 60 + f"\n{text}\n" + "=" * 60)


def main():
    # 1. data -----------------------------------------------------------
    if not os.path.exists(DATA):
        banner("STEP 1: Generating synthetic dataset")
        subprocess.run([sys.executable, os.path.join(BASE, "data", "generate_data.py")], check=True)

    # 2. cleaning -------------------------------------------------------
    banner("STEP 2: Cleaning data")
    df, report = cleaning.clean_emails(DATA)
    for k, v in report.items():
        print(f"  {k:24s}: {v}")

    # 3. features + scoring --------------------------------------------
    banner("STEP 3: Feature extraction & rule-based scoring")
    scorer = features.SpamScorer()
    mat, records = scorer.feature_matrix(df)
    scores = scorer.score(mat)
    df["score"] = scores
    df["predicted"] = scorer.classify(scores)
    for name in features.FEATURE_ORDER:
        df[f"f_{name}"] = [r[name] for r in records]
    print("  Rule weights:", features.WEIGHTS)
    print(f"  Thresholds  : promotion >= {scorer.t_promo}, spam >= {scorer.t_spam}")

    # 4. accuracy -------------------------------------------------------
    banner("STEP 4: Evaluation against the 'label' column")
    acc, cm = features.accuracy_and_confusion(df["label"], df["predicted"])
    print(f"  ACCURACY: {acc * 100:.2f}%  on {len(df)} emails")
    print("\n  Confusion matrix (rows = true, cols = predicted)")
    print(pd.DataFrame(cm).T.to_string())

    # 5. analysis -------------------------------------------------------
    banner("STEP 5: Inbox analytics")
    print("\nScore statistics (NumPy):")
    for k, v in analysis.score_stats(scores).items():
        print(f"  {k:7s}: {v:.2f}")
    print("\nTop sender domains and spam %:")
    print(analysis.domain_summary(df))
    print("\nPredicted label by day of week (pivot table):")
    print(analysis.label_by_day(df))
    print("\nPredicted label by time of day:")
    print(analysis.label_by_hour_block(df))
    keywords = analysis.top_spam_keywords(df, 15)
    print("\nTop spam keywords (Counter):", keywords[:10])
    hour_series = analysis.spam_by_hour(df)

    # 6. outputs --------------------------------------------------------
    banner("STEP 6: Saving charts and cleaned data")
    os.makedirs(OUT, exist_ok=True)
    paths = visuals.make_all(df, keywords, hour_series, OUT, scorer.t_promo, scorer.t_spam)
    for p in paths:
        print("  saved", os.path.relpath(p, BASE))
    df.to_csv(os.path.join(OUT, "cleaned_emails.csv"), index=False)
    print("  saved outputs/cleaned_emails.csv")
    print(f"\nDone. Final accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
