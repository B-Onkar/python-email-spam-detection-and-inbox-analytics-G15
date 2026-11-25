"""
features.py  -  regex feature extraction + rule-based scoring (no ML libraries).

Data structures used:
  list   -> list of feature records, tokens
  tuple  -> (feature_name, value) pairs
  dict   -> WEIGHTS (rule weights)
  set    -> SUSPICIOUS_TLDS / SUSPICIOUS_DOMAINS / STOP_WORDS
  Counter-> keyword frequencies
  class  -> SpamScorer (bonus)
"""
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np

# ---- Regex patterns --------------------------------------------------
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MONEY_RE = re.compile(r"(?:₹|\$|Rs\.?\s?)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
URGENT_RE = re.compile(r"\b(WIN|WINNER|FREE|URGENT|click here|verify|PRIZE|lottery)\b", re.IGNORECASE)
CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
EXCL_RE = re.compile(r"!")
PROMO_RE = re.compile(r"\b(sale|off|discount|offers?|deals?|shipping|unsubscribe|newsletter|"
                      r"arrivals|shop|order|member|exclusive|buy)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z']{3,}")

# ---- Sets ------------------------------------------------------------
SUSPICIOUS_TLDS = {"xyz", "top", "click", "biz", "info"}
SUSPICIOUS_DOMAINS = {"bit.ly", "tinyurl.com"}
STOP_WORDS = {"http", "https", "www", "com", "the", "and", "you", "your", "for", "are", "with", "this", "that", "will",
              "our", "can", "has", "have", "not", "from", "all", "now", "get", "was",
              "please", "here", "let", "its", "but", "may", "any"}

# ---- Dictionary of rule weights --------------------------------------
WEIGHTS = {
    "urgent_words": 1.5,
    "suspicious_domain": 3.0,
    "money_amount": 0.4,
    "all_caps": 0.5,
    "exclamations": 0.4,
    "invalid_sender": 0.8,
    "links": 0.5,
    "promo_words": 0.8,
}
FEATURE_ORDER = list(WEIGHTS.keys())      # fixed order -> vector alignment

# caps so that one feature cannot dominate the score
CAPS_LIMIT = {"urgent_words": 4, "money_amount": 3, "all_caps": 5, "exclamations": 5, "links": 3, "promo_words": 3}

# thresholds: score >= T_SPAM -> spam ; T_PROMO <= score < T_SPAM -> promotion ; else ham
T_PROMO = 1.6
T_SPAM = 5.5


def get_domain(url: str) -> str:
    """Return the domain of a URL (without www.)."""
    if url.lower().startswith("www."):
        url = "http://" + url
    host = urlparse(url).netloc.lower().rstrip(".,!;:)")
    return host[4:] if host.startswith("www.") else host


def is_suspicious_domain(domain: str) -> bool:
    if not domain:
        return False
    tld = domain.rsplit(".", 1)[-1]
    return domain in SUSPICIOUS_DOMAINS or tld in SUSPICIOUS_TLDS


def extract_features(subject: str, body: str, sender: str, valid_sender: bool):
    """Return a list of (feature_name, value) tuples for one e-mail."""
    text = f"{subject} {body}"
    urls = URL_RE.findall(text)
    domains = [get_domain(u) for u in urls]
    sender_domain = sender.split("@")[-1].lower() if valid_sender else ""
    suspicious = any(is_suspicious_domain(d) for d in domains + [sender_domain])
    # invalid sender with a suspicious-looking domain part is also caught here
    if not valid_sender and "@" in sender:
        suspicious = suspicious or is_suspicious_domain(sender.split("@")[-1].lower())

    values = {
        "urgent_words": len(URGENT_RE.findall(text)),
        "suspicious_domain": int(suspicious),
        "money_amount": len(MONEY_RE.findall(text)),
        "all_caps": len(CAPS_RE.findall(text)),
        "exclamations": len(EXCL_RE.findall(text)),
        "invalid_sender": int(not valid_sender),
        "links": len(urls),
        "promo_words": len(PROMO_RE.findall(text)),
    }
    for k, limit in CAPS_LIMIT.items():
        values[k] = min(values[k], limit)
    return [(name, values[name]) for name in FEATURE_ORDER]


def tokenize(text: str):
    """Lower-case word list without stop-words."""
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOP_WORDS]


def keyword_counter(texts):
    """Counter of word frequencies over a list of texts."""
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c


class SpamScorer:
    """Weighted-rule scorer. score = features . weights  (NumPy dot product)."""

    def __init__(self, weights=None, t_promo=T_PROMO, t_spam=T_SPAM):
        self.weights = weights or WEIGHTS
        self.t_promo = t_promo
        self.t_spam = t_spam
        self.w_vec = np.array([self.weights[f] for f in FEATURE_ORDER], dtype=float)

    def feature_matrix(self, df):
        """Build an (n_emails x n_features) NumPy matrix + a features DataFrame-ready list."""
        records = [
            dict(extract_features(r.subject, r.body, r.sender, r.valid_sender))
            for r in df.itertuples()
        ]
        mat = np.array([[rec[f] for f in FEATURE_ORDER] for rec in records], dtype=float)
        return mat, records

    def score(self, mat):
        return mat @ self.w_vec            # same as np.dot(mat, w_vec)

    def classify(self, scores):
        return np.where(scores >= self.t_spam, "spam",
                        np.where(scores >= self.t_promo, "promotion", "ham"))


def accuracy_and_confusion(y_true, y_pred, classes=("ham", "promotion", "spam")):
    """Accuracy + hand-built confusion matrix (dict of dicts)."""
    y_true, y_pred = list(y_true), list(y_pred)
    cm = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        if t in cm:
            cm[t][p] += 1
    correct = sum(cm[c][c] for c in classes)
    total = sum(sum(row.values()) for row in cm.values())
    return correct / total, cm
