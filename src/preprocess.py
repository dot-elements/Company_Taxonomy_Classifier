import re

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def stitch_company_text(row) -> str:
    parts = [
        row.get("description",""),
        str(row.get("business_tags","")).replace(",", " "),
        row.get("sector",""),
        row.get("category",""),
        row.get("niche",""),
    ]
    return normalize_text(" ".join([p for p in parts if p]))
