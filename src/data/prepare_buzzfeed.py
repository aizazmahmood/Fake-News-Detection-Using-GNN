# src/data/prepare_buzzfeed.py
import argparse, os, glob, xml.etree.ElementTree as ET, pandas as pd
from urllib.parse import urlparse
from src.utils.seed import set_seed

def parse_xml(path):
    tree = ET.parse(path); root = tree.getroot()
    def text(tag):
        el = root.find(tag)
        return (el.text or "").strip() if el is not None and el.text is not None else ""
    title = text("title")
    portal = text("portal")
    orientation = text("orientation")
    veracity = text("veracity")
    author = text("author")
    uri = text("uri")
    mainText = text("mainText")
    links = [a.get("href","").strip() for a in root.findall(".//hyperlink")]
    return dict(
        title=title, portal=portal, orientation=orientation, veracity=veracity,
        author=author, uri=uri, text=mainText, links="|||".join([l for l in links if l])
    )

def domain_from_uri(uri: str) -> str:
    try:
        dom = urlparse(uri).netloc.lower()
        return dom
    except Exception:
        return ""

def coalesce(a: pd.Series|None, b: pd.Series|None) -> pd.Series:
    """Prefer non-empty values from a, else b; treat '', 'nan', 'None' as missing."""
    if a is None and b is None:
        return pd.Series(dtype=object)
    if a is None:
        a = pd.Series([None]*len(b))
    if b is None:
        b = pd.Series([None]*len(a))
    a = a.astype(str)
    b = b.astype(str)
    def valid(s: pd.Series):
        s = s.fillna("").astype(str).str.strip()
        return (s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none") & (s.str.lower() != "null")
    out = b.copy()
    mask = valid(a)
    out.loc[mask] = a.loc[mask]
    # normalize empties to ""
    out = out.fillna("").astype(str).str.strip()
    return out

def normalize_veracity(v: pd.Series) -> pd.Series:
    """Lowercase + map a few common variants into the 4 canonical labels."""
    v = v.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "mostly true":"mostly true",
        "mostly_true":"mostly true",
        "true":"mostly true",
        "mixture of true and false":"mixture of true and false",
        "mixture":"mixture of true and false",
        "mixed":"mixture of true and false",
        "mostly false":"mostly false",
        "mostly_false":"mostly false",
        "false":"mostly false",
        "no factual content":"no factual content",
        "no-factual-content":"no factual content",
        "nofactualcontent":"no factual content",
    }
    v = v.map(lambda x: mapping.get(x, x))
    return v

def count_links(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    return s.apply(lambda x: 0 if x=="" else len([p for p in x.split("|||") if p.strip()]))

def main(in_dir, overview_csv, out_csv):
    set_seed(42)

    # Parse XMLs
    rows = []
    for fp in sorted(glob.glob(os.path.join(in_dir, "*.xml"))):
        rec = parse_xml(fp)
        rec["id"] = os.path.splitext(os.path.basename(fp))[0]  # e.g., '0001'
        rec["source_domain"] = domain_from_uri(rec["uri"])
        rows.append(rec)
    df = pd.DataFrame(rows)

    # Optional merge with overview; coalesce duplicate semantics
    if overview_csv and os.path.exists(overview_csv):
        ov = pd.read_csv(overview_csv)
        if "XML" in ov.columns:
            ov["id"] = ov["XML"].astype(str).str.zfill(4)
        # lowercase headers for safety
        ov.columns = [c.strip().lower() for c in ov.columns]
        # drop the original 'xml' column if present
        drop_ov = [c for c in ["xml"] if c in ov.columns]
        df = df.merge(ov.drop(columns=drop_ov), on="id", how="left", suffixes=("_x","_y"))

        for col in ["portal","orientation","veracity"]:
            x = f"{col}_x" if f"{col}_x" in df.columns else None
            y = f"{col}_y" if f"{col}_y" in df.columns else None
            if x or y:
                df[col] = coalesce(df.get(x), df.get(y))

        # coalesce uri/url if overview has url
        url_col = "url" if "url" in df.columns else None
        df["uri"] = coalesce(df.get("uri"), df.get(url_col))

        # drop the *_x/*_y leftovers (and url if we created it)
        drop_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        if url_col:
            drop_cols.append(url_col)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ---------- Cleaning / Normalization ----------
    # trim all string cols
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # lower-case certain categorical fields
    for c in ["portal","orientation"]:
        if c in df.columns:
            df[c] = df[c].str.lower()

    # veracity normalization
    if "veracity" in df.columns:
        df["veracity"] = normalize_veracity(df["veracity"])

    # ensure text availability: fallback to title when empty
    if "text" in df.columns:
        empty_text = df["text"].fillna("").astype(str).str.len() == 0
        if "title" in df.columns:
            df.loc[empty_text, "text"] = df.loc[empty_text, "title"].fillna("").astype(str)
        else:
            df.loc[empty_text, "text"] = ""

    # re-derive source_domain if empty
    if "source_domain" in df.columns:
        sd_empty = df["source_domain"].fillna("").eq("")
        df.loc[sd_empty, "source_domain"] = df.loc[sd_empty, "uri"].apply(domain_from_uri)
        df["source_domain"] = df["source_domain"].fillna("").str.lower()

    # add link count feature
    if "links" in df.columns:
        df["num_links"] = count_links(df["links"])

    # drop duplicate ids (keep first)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")

    # Final schema/order (include num_links if present)
    base_cols = ["id","title","text","portal","orientation","veracity","author","uri","source_domain","links"]
    if "num_links" in df.columns and "num_links" not in base_cols:
        base_cols.append("num_links")
    cols = [c for c in base_cols if c in df.columns]
    df = df[cols]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    # tiny summary to stdout for sanity
    print(f"Saved {len(df)} rows → {out_csv}")
    if "veracity" in df.columns:
        print("Veracity distribution:")
        print(df["veracity"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw/buzzfeed/articles")
    ap.add_argument("--overview_csv", default="data/raw/buzzfeed/overview.csv")
    ap.add_argument("--out_csv", default="data/processed/buzzfeed_all.csv")
    args = ap.parse_args()
    main(args.in_dir, args.overview_csv, args.out_csv)