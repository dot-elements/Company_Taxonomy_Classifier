from rank_bm25 import BM25Okapi
from preprocess import normalize_text

def tokenize(s: str) -> list[str]:
    return normalize_text(s).split()

class BM25LabelIndex:
    def __init__(self, label_docs: list[str], labels: list[str]):
        self.labels = labels
        corpus = [tokenize(doc) for doc in label_docs]
        self.bm25 = BM25Okapi(corpus)

    def topk(self, query: str, k: int = 15):
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        idx = scores.argsort()[::-1][:k]
        return [(self.labels[i], float(scores[i])) for i in idx]