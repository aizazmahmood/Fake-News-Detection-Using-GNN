import os, argparse, json, math, pandas as pd, numpy as np, torch
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict, Counter

from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors

# optional deps
try:
    import tldextract
    def etld1(u):
        try:
            ex = tldextract.extract(u)
            return ".".join([p for p in [ex.domain, ex.suffix] if p])
        except:
            return urlparse(u).netloc.lower()
except Exception:
    def etld1(u):
        try: return urlparse(u).netloc.lower()
        except: return ""

# text encoder
from sentence_transformers import SentenceTransformer
# optional NER (skip gracefully if not available)
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def embed_text(texts, model_name="sentence-transformers/all-mpnet-base-v2", bs=64):
    model = SentenceTransformer(model_name)
    embs = []
    for i in range(0, len(texts), bs):
        embs.append(model.encode(texts[i:i+bs], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
    return np.vstack(embs).astype(np.float32)

def extract_entities(text, top_k=10):
    if _nlp is None or not isinstance(text, str) or not text:
        return []
    doc = _nlp(text[:2000])
    keep = {"PERSON","ORG","GPE","NORP","EVENT","WORK_OF_ART"}
    ents = [e.text.strip() for e in doc.ents if e.label_ in keep and len(e.text.strip())>1]
    # prune by frequency within-article
    freq = Counter(ents)
    return [w for w,_ in freq.most_common(top_k)]

def parse_link_domains(links_col):
    out = []
    if isinstance(links_col, str) and links_col:
        for raw in links_col.split("|||"):
            dom = etld1(raw.strip())
            if dom: out.append(dom)
    return list(sorted(set(out)))  # unique per article

def main(args):
    ensure_dir(os.path.dirname(args.out_path))

    # load data
    bf = pd.read_csv(args.buzzfeed_csv)
    tr = pd.read_csv(os.path.join(args.split_dir, "train.csv"))
    dv = pd.read_csv(os.path.join(args.split_dir, "dev.csv"))
    te = pd.read_csv(os.path.join(args.split_dir, "test.csv"))
    label_map = {l:i for i,l in enumerate(sorted(bf["veracity"].astype(str).unique()))}
    num_classes = len(label_map)

    # align order & ids
    bf["id"] = bf["id"].astype(str)
    bf["text"] = bf["text"].fillna("").astype(str)
    bf["title"] = bf["title"].fillna("").astype(str)
    bf["num_links"] = bf.get("num_links", 0)

    # article node index
    art_ids = bf["id"].tolist()
    art_index = {aid:i for i,aid in enumerate(art_ids)}

    # features: article text embeddings + num_links scalar
    print("Embedding article texts …")
    Xtxt = embed_text(bf["text"].tolist(), model_name=args.model_name, bs=args.batch_size)
    num_links = bf["num_links"].fillna(0).astype(float).to_numpy()[:,None]
    Xart = np.hstack([Xtxt, num_links]).astype(np.float32)

    # domains from source_domain (fallback to uri)
    domains = []
    for u, sd in zip(bf.get("uri","").fillna(""), bf.get("source_domain","").fillna("")):
        d = sd if isinstance(sd,str) and sd else etld1(u)
        domains.append(d if d else "unknown")
    dom_vocab = {d:i for i,d in enumerate(sorted(set(domains)))}
    dom_index = [dom_vocab[d] for d in domains]

    # hosts from outgoing links
    host_lists = bf["links"].fillna("").apply(parse_link_domains).tolist()
    host_set = set()
    for lst in host_lists: host_set.update(lst)
    host_vocab = {h:i for i,h in enumerate(sorted(host_set))}
    # entity nodes (optional)
    ent_lists = []
    if args.with_entities:
        print("Extracting NER entities …")
        for t in bf["text"].tolist():
            ent_lists.append(extract_entities(t, top_k=8))
    else:
        ent_lists = [[] for _ in range(len(bf))]
    ent_set = set(e for lst in ent_lists for e in lst)
    ent_vocab = {e:i for i,e in enumerate(sorted(ent_set))}

    # topic nodes (optional, placeholder for later)
    # we’ll skip heavy BERTopic right now

    # build HeteroData
    data = HeteroData()
    data["article"].x = torch.tensor(Xart)                     # [N_art, 768+1]
    data["article"].y = torch.tensor([label_map[v] for v in bf["veracity"].astype(str)], dtype=torch.long)

    # domain nodes: aggregate article embeddings
    D = len(dom_vocab)
    dom_feats = np.zeros((D, Xart.shape[1]), dtype=np.float32)
    dom_count = np.zeros(D, dtype=np.int64)
    for i,di in enumerate(dom_index):
        dom_feats[di] += Xart[i]
        dom_count[di] += 1
    dom_count[dom_count==0] = 1
    dom_feats = dom_feats / dom_count[:,None]
    data["domain"].x = torch.tensor(dom_feats)

    # host nodes: aggregate from links
    H = len(host_vocab)
    if H > 0:
        host_feats = np.zeros((H, Xart.shape[1]), dtype=np.float32)
        host_count = np.zeros(H, dtype=np.int64)
        for i,lst in enumerate(host_lists):
            for h in lst:
                hi = host_vocab[h]
                host_feats[hi] += Xart[i]
                host_count[hi] += 1
        host_count[host_count==0] = 1
        host_feats = host_feats / host_count[:,None]
        data["host"].x = torch.tensor(host_feats)
    else:
        data["host"].x = torch.zeros((1, Xart.shape[1]))

    # entity nodes: embed entity strings (optional)
    if ent_vocab:
        print("Embedding entity strings …")
        ent_emb = embed_text(list(ent_vocab.keys()), model_name=args.model_name, bs=args.batch_size)
        data["entity"].x = torch.tensor(ent_emb)
    else:
        data["entity"].x = torch.zeros((1, Xart.shape[1]))

    # edges: article->domain (belongs_to)
    src = torch.tensor(range(len(bf)), dtype=torch.long)
    dst = torch.tensor(dom_index, dtype=torch.long)
    data["article", "belongs_to", "domain"].edge_index = torch.stack([src, dst], dim=0)
    data["domain", "rev_belongs_to", "article"].edge_index = torch.stack([dst, src], dim=0)

    # edges: article->host (links_to)
    a_src, h_dst = [], []
    for i,lst in enumerate(host_lists):
        for h in lst:
            a_src.append(i); h_dst.append(host_vocab[h])
    if len(a_src) == 0:
        a_src, h_dst = [0], [0]
    data["article","links_to","host"].edge_index = torch.tensor([a_src,h_dst], dtype=torch.long)
    data["host","rev_links_to","article"].edge_index = torch.tensor([h_dst,a_src], dtype=torch.long)

    # edges: article->entity (has_entity)
    e_src, e_dst = [], []
    for i,lst in enumerate(ent_lists):
        for e in lst:
            e_src.append(i); e_dst.append(ent_vocab[e])
    if len(e_src) == 0:
        e_src, e_dst = [0],[0]
    data["article","has_entity","entity"].edge_index = torch.tensor([e_src,e_dst], dtype=torch.long)
    data["entity","rev_has_entity","article"].edge_index = torch.tensor([e_dst,e_src], dtype=torch.long)

    # edges: article<->article similarity (top-k cosine)
    print("Building article similarity edges …")
    nn = NearestNeighbors(n_neighbors=args.topk+1, metric="cosine").fit(Xtxt)
    dists, idxs = nn.kneighbors(Xtxt)  # cosine distance ∈ [0,2]
    sim_src, sim_dst = [], []
    for i,row in enumerate(idxs):
        for j,nei in enumerate(row[1:]):  # skip self
            sim_src.append(i); sim_dst.append(int(nei))
    data["article","similar","article"].edge_index = torch.tensor([sim_src,sim_dst], dtype=torch.long)

    # masks from splits
    idx_train = torch.tensor([art_index[i] for i in tr["id"].astype(str)], dtype=torch.long)
    idx_dev   = torch.tensor([art_index[i] for i in dv["id"].astype(str)], dtype=torch.long)
    idx_test  = torch.tensor([art_index[i] for i in te["id"].astype(str)], dtype=torch.long)

    n = len(bf)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[idx_train] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[idx_dev] = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[idx_test] = True

    data["article"].train_mask = train_mask
    data["article"].val_mask   = val_mask
    data["article"].test_mask  = test_mask

    # metadata for later
    meta = {
        "label_map": label_map,
        "num_classes": num_classes,
        "node_dims": {nt: int(data[nt].x.shape[1]) for nt in data.node_types},
        "counts": {nt: int(data[nt].x.shape[0]) for nt in data.node_types},
        "edge_types": [str(et) for et in data.edge_types]
    }
    torch.save(data, args.out_path)
    with open(args.out_path.replace(".pt",".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved graph → {args.out_path}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buzzfeed_csv", default="data/processed/buzzfeed_all.csv")
    ap.add_argument("--split_dir",    default="data/processed/buzzfeed_split")
    ap.add_argument("--out_path",     default="data/processed/graphs/buzzfeed_hetero.pt")
    ap.add_argument("--model_name",   default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--batch_size",   type=int, default=64)
    ap.add_argument("--topk",         type=int, default=5)
    ap.add_argument("--with_entities", action="store_true")
    args = ap.parse_args()
    main(args)
