def explain_row(text, label, rule_c, bm25_c, dense_c, rerank_c, tx_synonyms):
    return {
        "label": label,
        "cross_encoder": round(rerank_c.get(label, 0.0), 3),
        "rule_hit": 1 if label in rule_c else 0,
        "bm25_score": round(bm25_c.get(label, 0.0), 3),
        "dense_sim": round(dense_c.get(label, 0.0), 3),
        "synonyms_seen": [w for w in tx_synonyms.get(label, []) if w in text]
    }