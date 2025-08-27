# pseudo main.py (just for your first run)
from data import load_companies, load_taxonomy
from preprocess import stitch_company_text, bm25_tokenize, shorten
from taxonomy import build_taxonomy_prompts
from rules import rule_candidates
from sparse import BM25LabelIndex
from dense import DenseLabelIndex
from rerank import LabelReranker, rerank
from combine import union_candidates, decide_labels
from explain import explain_row
import pandas as pd
import yaml

df = load_companies("../data/ml_insurance_challenge.csv").head(100)
df["text"] = df.apply(stitch_company_text, axis=1)
df["sector"] = df["sector"].fillna("").astype(str).str.lower().replace("nan", "")
tx = build_taxonomy_prompts(load_taxonomy("../data/insurance_taxonomy.csv"))
tx_labels = tx["label"].tolist()
tx_prompts = tx["prompt"].tolist()
tx_syn = dict(zip(tx["label"], tx["synonyms"]))

print(tx.head())
tx_labels = tx["label"].astype(str).str.strip().tolist()
assert len(tx_labels) > 0, "No taxonomy labels?"
print("Unique labels:", len(set(tx_labels)))



# 1) candidates
rule_c = rule_candidates(df, set(tx_labels))
bm25 = BM25LabelIndex([ " ".join(bm25_tokenize(p)) for p in tx_prompts ], tx_labels)
bm25_c = [ dict(bm25.topk(" ".join(bm25_tokenize(t)), k=15)) for t in df["text"]]
# print('printing bm25_c')
# print(bm25_c)
dense = DenseLabelIndex(tx_prompts, tx_labels, model_name="sentence-transformers/all-mpnet-base-v2")
dense_inputs = [shorten(t, 1000) for t in df["text"]]
dense_c = dense.topk(dense_inputs, k=15); dense_c = [ dict(x) for x in dense_c ]

cand = union_candidates(rule_c, bm25_c, dense_c, k=15)
# print('printing cand')
# print(cand)
# 2) rerank
reranker = LabelReranker()
rerank_c = rerank(df["text"].tolist(), tx_prompts, tx_labels, cand, reranker)
print('printing rerank_c')
print(rerank_c)
# 3) decisions + explanations
outs = []
for i, row in df.iterrows():
    chosen = decide_labels(rerank_c[i], rule_c[i], row.get("sector",""), global_tau=0.0, max_labels=3)
    exps = [explain_row(row["text"], lab, rule_c[i], bm25_c[i], dense_c[i], rerank_c[i], tx_syn) for lab in chosen]
    outs.append({"insurance_label": chosen, "explanations": exps})

res = pd.concat(
    [df.drop(columns=["text"]).reset_index(drop=True), pd.DataFrame(outs)],
    axis=1
)
res.to_json("../outputs/annotated_debug.json", orient="records", lines=True)
