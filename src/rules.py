from preprocess import normalize_text

# quick rule table: substring -> label
RULES = [
    ("automotive body", "Auto Body Shops"),
    ("collision repair", "Auto Body Shops"),
    ("windshield", "Auto Glass Repair"),
    ("auto glass", "Auto Glass Repair"),
    ("roofing", "Roofing Contractors"),
]

def rule_candidates(df, labels_set):
    hits = []
    for i, row in df.iterrows():
        txt = normalize_text(" ".join([str(row.get("niche","")), str(row.get("category","")), str(row.get("business_tags",""))]))
        lab_scores = {}
        for needle, label in RULES:
            if needle in txt and label in labels_set:
                lab_scores[label] = max(lab_scores.get(label, 0.0), 1.0)  # 1.0 = rule precision anchor
        hits.append(lab_scores)
    return hits  # list[dict[label -> score]]
