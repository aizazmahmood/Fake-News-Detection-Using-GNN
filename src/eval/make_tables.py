import argparse, json, re
from pathlib import Path
import pandas as pd
from io import StringIO

ap = argparse.ArgumentParser()
ap.add_argument("--hf_dev", type=str, default="")
ap.add_argument("--hf_test", type=str, default="")
ap.add_argument("--gnn_dev", type=str, required=True)
ap.add_argument("--gnn_test", type=str, required=True)
ap.add_argument("--ens_dev", type=str, required=True)
ap.add_argument("--ens_test", type=str, required=True)
ap.add_argument("--out_dir", type=str, default="outputs/final")
args = ap.parse_args()

def find_report_file(p: str) -> Path:
    """If p is a directory, try report.json then report.txt. If p is a file, return it."""
    path = Path(p)
    if path.is_file():
        return path
    if path.is_dir():
        for name in ["report.json", "report.txt"]:
            cand = path / name
            if cand.exists():
                return cand
        # otherwise pick first matching file
        for ext in ("*.json", "*.txt"):
            files = list(path.glob(ext))
            if files:
                return files[0]
    raise FileNotFoundError(f"Could not locate a report file under: {p}")

def load_json(path: Path):
    d = json.loads(path.read_text())
    # expect sklearn classification_report dict shape
    acc = float(d.get("accuracy", 0.0))
    macro = float(d["macro avg"]["f1-score"])
    per = {k: v["f1-score"] for k, v in d.items()
           if k not in ["accuracy", "macro avg", "weighted avg"]}
    return acc, macro, per

def parse_txt_report(txt: str):
    """
    Parse sklearn classification_report plain text.
    Handles class labels with spaces (e.g., 'mixture of true and false').
    """
    lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
    # grab table lines after the header
    # header usually contains 'precision    recall    f1-score   support'
    start = 0
    for i, ln in enumerate(lines):
        if re.search(r"\bprecision\b\s+\brecall\b\s+\bf1-score\b\s+\bsupport\b", ln):
            start = i + 1
            break
    rows = []
    for ln in lines[start:]:
        # stop if a separator like 'micro avg' or something weird appears (rare here)
        # We accept rows like:
        # label .... precision  recall  f1-score  support
        m_acc = re.match(r"^\s*accuracy\s+([0-9.]+)(?:\s+\d+)?\s*$", ln, re.IGNORECASE)
        if m_acc:
            rows.append(("accuracy", None, None, float(m_acc.group(1)), 0))
            continue
        m = re.match(r"(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", ln)
        if m:
            label = m.group(1).strip()
            prec, rec, f1 = [float(x) for x in m.groups()[1:4]]
            sup = int(m.group(5))
            rows.append((label, prec, rec, f1, sup))
        else:
            # accuracy row looks like: 'accuracy                         0.7714       245'
            m2 = re.match(r"(accuracy)\s+([0-9.]+)\s+(\d+)$", ln)
            if m2:
                rows.append((m2.group(1), None, None, float(m2.group(2)), int(m2.group(3))))
    if not rows:
        # fallback: try fixed-width parsing
        df = pd.read_fwf(StringIO("\n".join(lines[start:])))
        df = df.dropna(how="all", axis=1).dropna(how="all")
        df.columns = [c.strip() or f"col{i}" for i,c in enumerate(df.columns)]
        # attempt to normalize
        out = {}
        acc = 0.0
        macro = 0.0
        for _, r in df.iterrows():
            label = str(r.iloc[0]).strip()
            if label.lower() == "accuracy":
                try: acc = float(r[-2])
                except: pass
            else:
                try:
                    f1 = float(r.get("f1-score", r.iloc[-2]))
                    out[label] = f1
                except: pass
        macro = float(out.get("macro avg", 0.0))
        # Remove macro/weighted from per-class
        out.pop("macro avg", None); out.pop("weighted avg", None)
        return acc, macro, out

    # Build dicts
    acc = 0.0
    per = {}
    macro = 0.0
    for label, prec, rec, f1, sup in rows:
        if label == "accuracy":
            acc = f1  # stored in f1 slot by our regex
        elif label in ("macro avg", "weighted avg"):
            if label == "macro avg": macro = f1
        else:
            per[label] = f1
    return acc, macro, per

def load_any(p: str):
    rp = find_report_file(p)
    if rp.suffix.lower() == ".json":
        return load_json(rp)
    elif rp.suffix.lower() == ".txt":
        return parse_txt_report(rp.read_text())
    else:
        raise ValueError(f"Unsupported report type: {rp}")

def add_row(rows, name, dev_path, test_path):
    acc_d, macro_d, per_d = load_any(dev_path)
    acc_t, macro_t, per_t = load_any(test_path)
    rows.append({
        "Model": name,
        "Dev Acc": f"{acc_d:.3f}",
        "Dev Macro-F1": f"{macro_d:.3f}",
        "Test Acc": f"{acc_t:.3f}",
        "Test Macro-F1": f"{macro_t:.3f}",
    })
    return per_d, per_t

out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

rows = []

# Optional HF if you have it:
if args.hf_dev and args.hf_test:
    add_row(rows, "HF (MPNet)", args.hf_dev, args.hf_test)

per_gnn_d, per_gnn_t = add_row(rows, "GNN (HeteroSAGE, focal)",
                               args.gnn_dev, args.gnn_test)

per_ens_d, per_ens_t = add_row(rows, "Ensemble (alpha=0.70)",
                               args.ens_dev, args.ens_test)

df = pd.DataFrame(rows)
(out_dir / "results.md").write_text(df.to_markdown(index=False), encoding="utf-8")
(out_dir / "results.tex").write_text(df.to_latex(index=False, escape=True), encoding="utf-8")

per_test = pd.DataFrame({
    "GNN": per_gnn_t,
    "Ensemble (alpha=0.70)": per_ens_t,
}).T
(out_dir / "per_class_test.md").write_text(per_test.to_markdown())

print("Saved tables →", out_dir)
