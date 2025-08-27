import re

import pandas as pd


def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()
def _to_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        print('NAN value!')
        return ""
    if isinstance(x, (list, tuple, set)):
        return " ".join(_to_text(i) for i in x if _to_text(i))
    return str(x)
def stitch_company_text(row) -> str:
    parts = [
        row.get("description",""),
        str(row.get("business_tags","")).strip("[]").replace("'", "").replace(",", " "),
        row.get("sector",""),
        row.get("category",""),
        row.get("niche",""),
    ]
    clean = [_to_text(p) for p in parts if _to_text(p)]
    return normalize_text(" ".join(clean))

def load_stopwords(path="../configs/domain_stopwords.txt"):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def bm25_tokenize(s, path_stopwords="../configs/domain_stopwords.txt"):
    domain_stop = load_stopwords(path_stopwords)
    if not isinstance(s, str): return []
    s = s.lower()
    # strip urls/emails/phones (noise for lexical match)
    s = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\+?\d[\d\s\-()]{6,}', ' ', s)
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t not in domain_stop and len(t) > 2]

def clean_for_encoder(s,n=800):
    if not isinstance(s, str): return ""
    s = s.strip()
    # Drop obvious boilerplate that wastes tokens
    s = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', ' ', s)   # urls/emails
    s = re.sub(r'\+?\d[\d\s\-()]{6,}', ' ', s)             # phones
    s = re.sub(r'\b(inc|ltd|llc|gmbh|s\.a\.|srl|pte|bv)\b\.?', ' ', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s)
    return shorten(s,n)
def shorten(s, n=800):
    s = s.strip()
    return s if len(s) <= n else s[:n]