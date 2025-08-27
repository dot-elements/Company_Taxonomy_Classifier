def union_candidates(rule_c, bm25_c, dense_c, k=15, w_rule=1.0, w_bm25=0.2, w_dense=0.3):
    """
    rule_c, bm25_c, dense_c are dict[label->raw_score] per row.
    Returns dict[label->combined_score] per row, top-K kept.
    """
    out = []
    for rc, sc, dc in zip(rule_c, bm25_c, dense_c):
        scores = {}
        for lab, v in rc.items(): scores[lab] = scores.get(lab, 0.0) + w_rule * v
        for lab, v in sc.items(): scores[lab] = scores.get(lab, 0.0) + w_bm25 * v
        for lab, v in dc.items(): scores[lab] = scores.get(lab, 0.0) + w_dense * v
        # top-K
        top = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k])
        out.append(top)
    return out