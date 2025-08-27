import yaml
from preprocess import normalize_text

def make_label_prompt(label: str, synonyms: list[str]|None=None) -> str:
    syn = ", ".join(synonyms or [])
    base = f"{label}. keywords: {syn}" if syn else label
    return normalize_text(base)

def build_taxonomy_prompts(tx_df, synonyms_yaml="configs/synonyms.yaml"):
    try:
        syn = yaml.safe_load(open(synonyms_yaml))  # dict: {label: [syn1, syn2, ...]}
    except FileNotFoundError:
        syn = {}

    tx_df = tx_df.copy()
    tx_df["synonyms"] = tx_df["label"].map(lambda l: syn.get(l, []))
    tx_df["prompt"] = tx_df.apply(lambda r: make_label_prompt(r["label"], r["synonyms"]), axis=1)
    return tx_df
