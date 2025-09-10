# src/data/clean_fnn.py
import argparse, os, pandas as pd
from urllib.parse import urlparse

def domain_from_url(url):
    try: return urlparse(url).netloc.lower()
    except: return ""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/processed/fnn_all.csv")
    ap.add_argument("--out_csv", default="data/processed/fnn_all_clean.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    # re-fill domain if possible
    mask = df["source_domain"].isna() | (df["source_domain"]=="")
    df.loc[mask, "source_domain"] = df.loc[mask, "news_url"].fillna("").apply(domain_from_url)
    df["source_domain"] = df["source_domain"].replace("", "unknown").fillna("unknown")

    # drop ids with conflicting labels
    bad_ids = df.groupby("id")["label"].nunique()
    bad_ids = bad_ids[bad_ids>1].index.tolist()
    if bad_ids:
        before = len(df)
        df = df[~df["id"].isin(bad_ids)].copy()
        print(f"Dropped {before-len(df)} rows with conflicting labels: {bad_ids}")

    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows → {args.out_csv}")