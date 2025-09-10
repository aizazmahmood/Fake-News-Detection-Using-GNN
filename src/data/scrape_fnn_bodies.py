import argparse, os, asyncio, httpx, pandas as pd, trafilatura
from urllib.parse import urlparse
from tqdm import tqdm

def clean_url(u): 
    try: return u.strip()
    except: return ""

async def fetch(session, url, timeout=15.0):
    try:
        r = await session.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0 (FND/1.0)"})
        if r.status_code==200:
            return r.text
    except Exception:
        return None
    return None

async def run(urls, max_conc=10):
    out = []
    limits = httpx.Limits(max_keepalive_connections=max_conc, max_connections=max_conc)
    async with httpx.AsyncClient(limits=limits, verify=False) as session:
        for u in tqdm(urls):
            html = await fetch(session, u)
            text = trafilatura.extract(html) if html else None
            out.append(text or "")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/processed/fnn_all_clean.csv")
    ap.add_argument("--out_csv", default="data/processed/fnn_all_with_text.csv")
    ap.add_argument("--url_col", default="news_url")
    ap.add_argument("--max_conc", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    urls = [clean_url(u) for u in df[args.url_col].fillna("").astype(str).tolist()]
    texts = asyncio.run(run(urls, args.max_conc))
    df["body_text"] = texts
    # convenience: title+body
    df["title_plus_body"] = (df["title"].fillna("") + ". " + df["body_text"].fillna("")).str.strip()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows → {args.out_csv}")
