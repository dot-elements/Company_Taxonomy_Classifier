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

def decide_labels(rerank_scores: dict[str,float],
                  rule_hits: dict[str,float],
                  sector: str,
                  global_tau=0.6,
                  max_labels=3):
    # sector gating example (toy)
    blocked = set()
    if "health" in sector:
        blocked.update({"Auto Body Shops","Auto Glass Repair"})
    # priority: rules first
    chosen = []
    # 1) take rule-backed labels that exceed (global_tau - 0.1) or always if rule fires
    for lab, s in sorted(rerank_scores.items(), key=lambda x: x[1], reverse=True):
        if lab in blocked: continue
        if lab in rule_hits or s >= global_tau:
            chosen.append((lab, s))
        if len(chosen) >= max_labels: break
    return [lab for lab, _ in chosen]