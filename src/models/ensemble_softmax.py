# src/models/ensemble_softmax.py
import argparse, json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_geometric.data import HeteroData
from src.models.train_hetero_gnn import HeteroSAGE

def hf_probs(model_dir, texts, max_len=256):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    id2label_hf = (mdl.config.id2label if hasattr(mdl.config, "id2label") else
                   {i: str(i) for i in range(mdl.config.num_labels)})
    with torch.no_grad():
        out=[]
        for t in texts:
            enc = tok((t or ""), truncation=True, max_length=max_len, return_tensors="pt")
            p = torch.softmax(mdl(**enc).logits, dim=1).numpy()[0]
            out.append(p)
    # normalize keys to int
    id2label_hf = {int(k.split("_")[-1]) if isinstance(k,str) else int(k): v for k,v in id2label_hf.items()}
    return np.array(out), id2label_hf  # [N,C]

def gnn_probs(model_path, graph_path):
    data: HeteroData = torch.load(graph_path, map_location="cpu")
    raw_in = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
    num_classes = int(data["article"].y.max().item()) + 1
    mdl = HeteroSAGE(data.metadata(), raw_in, hidden=64, out_classes=num_classes)
    mdl.load_state_dict(torch.load(model_path, map_location="cpu"))
    mdl.eval()
    with torch.no_grad():
        logits = mdl({k:v.float() for k,v in data.x_dict.items()}, data.edge_index_dict)
        P = torch.softmax(logits, dim=1).numpy()

    meta = json.load(open(graph_path.replace(".pt",".json")))
    label2id_g = meta["label_map"]                # name -> id
    id2label_g = {v:k for k,v in label2id_g.items()}  # id -> name
    return P, id2label_g

def reorder_cols(P_src, id2label_src, id2label_tgt):
    name2idx_src = {name:i for i,name in id2label_src.items()}
    idx = [name2idx_src[id2label_tgt[i]] for i in range(len(id2label_tgt))]
    return P_src[:, idx]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buzzfeed_csv", required=True)
    ap.add_argument("--model_dir_hf", required=True)
    ap.add_argument("--graph_pt", required=True)
    ap.add_argument("--gnn_weights", required=True)
    ap.add_argument("--split_csv", required=True)   # dev/test with an "id" column
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.6)  # HF weight
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--text_col", default="text")
    args = ap.parse_args()

    # --- Load & normalize IDs as STR everywhere ---
    df_all = pd.read_csv(args.buzzfeed_csv)
    df_all[args.id_col] = df_all[args.id_col].astype(str).str.strip()
    df_all = df_all.set_index(args.id_col)
    ids = pd.read_csv(args.split_csv)[args.id_col].astype(str).str.strip().tolist()

    # Sanity: anything missing?
    missing = [i for i in ids if i not in df_all.index]
    if missing:
        raise SystemExit(f"{len(missing)} ids from split not found in buzzfeed_csv index. Example: {missing[:10]}")

    texts = df_all.loc[ids, args.text_col].fillna("").astype(str).tolist()

    # --- Probs ---
    P_hf, id2label_hf = hf_probs(args.model_dir_hf, texts)
    P_g,  id2label_g  = gnn_probs(args.gnn_weights, args.graph_pt)

    # Align graph to HF label order, and align row order to split ids
    idx_rows = [df_all.index.get_loc(i) for i in ids]   # safe: same str dtype now
    P_g = P_g[idx_rows, :]
    P_g = reorder_cols(P_g, id2label_g, id2label_hf)

    # Fuse
    P = args.alpha * P_hf + (1.0 - args.alpha) * P_g
    np.savetxt(args.out_csv, P, delimiter=",")
    print("Saved →", args.out_csv)
