from sentence_transformers import CrossEncoder
import numpy as np
from scipy.special import expit

class LabelReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def score_pairs(self, pairs, batch_size=512, return_proba=True):
        scores = np.array(self.model.predict(pairs, batch_size=batch_size))
        if return_proba:
            scores = expit(scores)
        return scores

def rerank(df_texts, tx_prompts, tx_labels, cand_dicts, reranker: LabelReranker):
    """
    For each row, build (text, prompt) pairs for its candidates; score; return dict[label->score].
    """
    out = []
    for text, cands in zip(df_texts, cand_dicts):
        labels = list(cands.keys())
        pairs = [(text, tx_prompts[tx_labels.index(lab)]) for lab in labels]
        scores = reranker.score_pairs(pairs)
        out.append(dict(zip(labels, scores.tolist())))
    return out