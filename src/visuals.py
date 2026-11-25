"""
visuals.py  -  Matplotlib charts saved as PNG in outputs/.
"""
import os

import matplotlib
matplotlib.use("Agg")           # no window needed; works on any machine
import matplotlib.pyplot as plt


def _save(fig_name, out_dir):
    path = os.path.join(out_dir, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def pie_spam_vs_ham(df, out_dir):
    counts = df["predicted"].value_counts().reindex(["ham", "promotion", "spam"], fill_value=0)
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
            colors=["#4caf50", "#ffb300", "#e53935"])
    plt.title("Inbox composition (predicted)")
    return _save("01_spam_vs_ham_pie.png", out_dir)


def bar_top_keywords(keywords, out_dir):
    words = [w for w, _ in keywords][::-1]
    freq = [c for _, c in keywords][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(words, freq, color="#e53935")
    plt.xlabel("Frequency")
    plt.ylabel("Keyword")
    plt.title("Top keywords in spam emails")
    return _save("02_top_spam_keywords.png", out_dir)


def hist_scores(df, out_dir, t_promo, t_spam):
    plt.figure(figsize=(8, 5))
    for lab, col in [("ham", "#4caf50"), ("promotion", "#ffb300"), ("spam", "#e53935")]:
        plt.hist(df.loc[df["label"] == lab, "score"], bins=20, alpha=0.6, label=lab, color=col)
    plt.axvline(t_promo, color="black", linestyle="--", label=f"promo threshold = {t_promo}")
    plt.axvline(t_spam, color="red", linestyle="--", label=f"spam threshold = {t_spam}")
    plt.xlabel("Spam score")
    plt.ylabel("Number of emails")
    plt.title("Distribution of spam scores by true label")
    plt.legend()
    return _save("03_spam_score_histogram.png", out_dir)


def line_spam_by_hour(hour_series, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(hour_series.index, hour_series.values, marker="o", color="#e53935")
    plt.xticks(range(0, 24, 2))
    plt.xlabel("Hour of day")
    plt.ylabel("Spam emails")
    plt.title("When does spam arrive?")
    plt.grid(alpha=0.3)
    return _save("04_spam_by_hour.png", out_dir)


def make_all(df, keywords, hour_series, out_dir, t_promo, t_spam):
    os.makedirs(out_dir, exist_ok=True)
    return [
        pie_spam_vs_ham(df, out_dir),
        bar_top_keywords(keywords, out_dir),
        hist_scores(df, out_dir, t_promo, t_spam),
        line_spam_by_hour(hour_series, out_dir),
    ]
