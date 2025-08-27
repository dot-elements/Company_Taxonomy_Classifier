import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class DenseLabelIndex:
    def __init__(self, label_docs: list[str], labels: list[str], model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.labels = labels
        self.lab_vecs = self.model.encode(label_docs, normalize_embeddings=True, batch_size=64)
        d = self.lab_vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cosine via inner product on normalized vecs
        self.index.add(self.lab_vecs.astype(np.float32))

    def topk(self, query_texts: list[str], k: int = 20):
        qv = self.model.encode(query_texts, normalize_embeddings=True, batch_size=64).astype(np.float32)
        sims, idxs = self.index.search(qv, k)
        out = []
        for sim_row, idx_row in zip(sims, idxs):
            out.append([(self.labels[j], float(s)) for j, s in zip(idx_row, sim_row)])
        return out