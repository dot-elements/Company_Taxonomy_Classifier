# src/data.py
import pandas as pd

def load_companies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize expected columns
    expected = ["name","description","business_tags","sector","category","niche"]
    missing = [c for c in expected if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    return df

def load_taxonomy(path: str) -> pd.DataFrame:
    tx = pd.read_csv(path)
    # expect a column 'label' (or rename)
    if "label" not in tx.columns:
        # try to infer
        first = tx.columns[0]
        tx = tx.rename(columns={first:"label"})
    tx["label"] = tx["label"].astype(str).str.strip()
    return tx
