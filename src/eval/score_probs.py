# src/eval/score_probs.py
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def load_labels(hf_dir: str):
    cfg = json.load(open(Path(hf_dir) / "config.json"))
    id2label = cfg.get("id2label") or {i: str(i) for i in range(cfg["num_labels"])}
    # normalize to consecutive int keys
    id2label = {int(k.split("_")[-1]) if isinstance(k, str) else int(k): v for k, v in id2label.items()}
    labels = [id2label[i] for i in range(len(id2label))]
    return labels

def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.hf_model_dir)
    name2id = {n: i for i, n in enumerate(labels)}

    split = pd.read_csv(args.split_csv)
    y_true = split[args.label_col].map(name2id).to_numpy()

    P = np.loadtxt(args.probs_csv, delimiter=",")
    assert P.shape[0] == len(split), f"rows mismatch: probs {P.shape[0]} vs split {len(split)}"
    assert P.shape[1] == len(labels), f"cols mismatch: probs {P.shape[1]} vs num labels {len(labels)}"

    y_pred = P.argmax(1)

    # reports
    rep_txt = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    rep_json = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")

    (out / "report.json").write_text(json.dumps(rep_json, indent=2))
    (out / "report.txt").write_text(rep_txt + f"\n\naccuracy={acc:.4f} macro_f1={macro:.4f}\n")

    # confusion (raw + normalized heatmap)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    np.savetxt(out / "confusion.csv", cm, delimiter=",", fmt="%d")

    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(6,5))
    sns.heatmap(cmn, annot=np.round(cmn, 2), fmt=".2f", cmap="viridis",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(out / "confusion.png", dpi=150); plt.close()

    print(rep_txt)
    print(f"accuracy={acc:.4f} macro_f1={macro:.4f}")
    print(f"Saved outputs → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--probs_csv", required=True)
    ap.add_argument("--hf_model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_col", default="veracity")
    main(ap.parse_args())
