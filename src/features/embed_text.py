import argparse, os, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def batch_encode(texts, model, bs=64):
    embs = []
    for i in tqdm(range(0, len(texts), bs)):
        embs.append(model.encode(
            texts[i:i+bs],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ))
    return np.vstack(embs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--model_name", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    texts = df[args.text_col].fillna("").astype(str).tolist()
    model = SentenceTransformer(args.model_name)
    X = batch_encode(texts, model, bs=64)
    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)
    np.save(args.out_npy, X)
    print(f"Saved embeddings: {X.shape} → {args.out_npy}")
