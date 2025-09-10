import argparse, os, pandas as pd
from urllib.parse import urlparse
from src.utils.seed import set_seed

def load_csv(path, label):
    df = pd.read_csv(path)
    df["label"] = label
    return df

def share_count_from_tweet_ids(series):
    def count(x):
        if isinstance(x, str) and len(x):
            return len([t for t in x.split("\t") if t])
        return 0
    return series.apply(count)

def domain_from_url(url):
    try: return urlparse(url).netloc.lower()
    except: return ""

def main(raw_dir, out_csv):
    set_seed(42)
    paths = {
        "politifact_real": ("politifact_real.csv", 1),
        "politifact_fake": ("politifact_fake.csv", 0),
        "gossipcop_real": ("gossipcop_real.csv", 1),
        "gossipcop_fake": ("gossipcop_fake.csv", 0),
        "fakenewsnet": ("FakeNewsNet.csv", None),
    }
    frames = []
    for _, (fname, lab) in paths.items():
        p = os.path.join(raw_dir, fname)
        if not os.path.exists(p): 
            continue
        df = pd.read_csv(p)
        if "tweet_ids" in df.columns:
            df["share_count"] = share_count_from_tweet_ids(df["tweet_ids"])
        elif "tweet_num" in df.columns:
            df["share_count"] = df["tweet_num"].fillna(0)
        else:
            df["share_count"] = 0
        if "label" not in df.columns and lab is not None:
            df["label"] = lab
        # Standardize cols
        std = pd.DataFrame({
            "id": df.get("id", pd.Series(range(len(df)))),
            "title": df.get("title", ""),
            "news_url": df.get("news_url", df.get("uri","")),
            "source_domain": df.get("source_domain", ""),
            "label": df.get("real", df.get("label", None)),
            "share_count": df["share_count"]
        })
        # fill domain if missing
        mask = std["source_domain"].isna() | (std["source_domain"]=="")
        std.loc[mask, "source_domain"] = std.loc[mask, "news_url"].apply(domain_from_url)
        frames.append(std)
    out = pd.concat(frames, ignore_index=True)
    out.dropna(subset=["title"], inplace=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows → {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/fnn")
    ap.add_argument("--out_csv", default="data/processed/fnn_all.csv")
    args = ap.parse_args()
    main(args.raw_dir, args.out_csv)
