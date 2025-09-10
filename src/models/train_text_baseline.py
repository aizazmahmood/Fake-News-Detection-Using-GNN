import argparse, os, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, auc)
from sklearn.feature_extraction import FeatureHasher
import matplotlib.pyplot as plt

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def load_emb(path):
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(f"Embeddings {path} need shape [N, D], got {X.shape}")
    return X

def hash_domains(domains, n_features=1024):
    # Represent each domain as a one-hot dict for the hasher: {"dom=example.com": 1}
    feats = [{"dom=" + (d if isinstance(d,str) else ""): 1} for d in domains]
    h = FeatureHasher(n_features=n_features, input_type="dict", alternate_sign=False)
    Xh = h.transform(feats).toarray().astype(np.float32)
    return Xh

def add_optional_features(df, X_base, extras):
    mats = [X_base]
    info = {}
    if "share_count_log" in extras and "share_count" in df.columns:
        sc = np.log1p(df["share_count"].fillna(0).astype(float).values).reshape(-1,1)
        mats.append(sc.astype(np.float32))
        info["share_count_log"] = True
    if "num_links" in extras and "num_links" in df.columns:
        nl = df["num_links"].fillna(0).astype(float).values.reshape(-1,1)
        mats.append(nl.astype(np.float32))
        info["num_links"] = True
    if "domain_hash" in extras and "source_domain" in df.columns:
        Xh = hash_domains(df["source_domain"].fillna("unknown").astype(str).values)
        mats.append(Xh)
        info["domain_hash"] = Xh.shape[1]
    X = np.hstack(mats).astype(np.float32)
    return X, info

def plot_confusion(cm, labels, out_png, normalize=True, title="Confusion Matrix"):
    fig = plt.figure(figsize=(6,5))
    cmn = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) if normalize else cm
    im = plt.imshow(cmn, interpolation='nearest')
    plt.title(title)
    plt.colorbar(im)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = cmn.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = format(cmn[i, j], fmt)
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if cmn[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_class_f1(report_dict, out_png, title="Class-wise F1"):
    labels = [k for k in report_dict.keys() if k not in ["accuracy","macro avg","weighted avg"]]
    f1s = [report_dict[k]["f1-score"] for k in labels]
    fig = plt.figure(figsize=(7,4))
    plt.bar(labels, f1s)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0,1.0)
    plt.title(title)
    plt.ylabel("F1-score")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_roc_pr(y_true, y_prob, out_dir):
    ensure_dir(out_dir)
    # ROC-AUC
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = None

    # PR curve
    try:
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
    except Exception:
        prec, rec, pr_auc = None, None, None

    # Plot ROC-like (we only have AUC number unless we compute fpr/tpr with sklearn.roc_curve)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc:.4f}" if roc is not None else "AUC=N/A")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close(fig)

    if prec is not None:
        fig = plt.figure(figsize=(6,5))
        plt.plot(rec, prec, label=f"AP={pr_auc:.4f}" if pr_auc is not None else "AP=N/A")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
        plt.legend()
        fig.savefig(os.path.join(out_dir, "pr_curve.png"), bbox_inches="tight")
        plt.close(fig)

    return {"roc_auc": roc, "pr_auc": pr_auc}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--train_emb", required=True)
    ap.add_argument("--dev_emb", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--out_dir", required=False, default="outputs/baselines")
    ap.add_argument("--extras", default="", help="comma-separated: share_count_log,num_links,domain_hash")
    ap.add_argument("--dataset_name", default="dataset")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    extras = [t.strip() for t in args.extras.split(",") if t.strip()]

    # Load
    tr_df = pd.read_csv(args.train_csv)
    dv_df = pd.read_csv(args.dev_csv)
    Xtr = load_emb(args.train_emb)
    Xdv = load_emb(args.dev_emb)

    # Labels
    y_tr_raw = tr_df[args.label_col].astype(str).fillna("")
    y_dv_raw = dv_df[args.label_col].astype(str).fillna("")
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    y_dv = le.transform(y_dv_raw)
    class_names = list(le.classes_)

    # Extras
    Xtr, info_tr = add_optional_features(tr_df, Xtr, extras)
    Xdv, info_dv = add_optional_features(dv_df, Xdv, extras)

    # Model
    clf = LogisticRegression(
        max_iter=3000, class_weight="balanced",
        solver="saga", penalty="l2", n_jobs=-1
    )
    clf.fit(Xtr, y_tr)
    dv_pred = clf.predict(Xdv)
    try:
        dv_prob = clf.predict_proba(Xdv)
    except Exception:
        dv_prob = None

    # Reports
    rep = classification_report(y_dv, dv_pred, target_names=class_names, output_dict=True, digits=4)
    rep_txt = classification_report(y_dv, dv_pred, target_names=class_names, digits=4)

    # Output directory per experiment
    feat_tag = "+".join(extras) if extras else "text_only"
    exp_dir = os.path.join(args.out_dir, f"{args.dataset_name}_lr_{feat_tag}")
    ensure_dir(exp_dir)

    # Save report
    with open(os.path.join(exp_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt + "\n")
    with open(os.path.join(exp_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_dv, dv_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, os.path.join(exp_dir, "confusion_matrix.png"), normalize=True)

    # Class-wise F1 bars (multi-class)
    if len(class_names) > 2:
        plot_class_f1(rep, os.path.join(exp_dir, "class_f1.png"))

    # Binary curves
    metrics_bin = {}
    if len(class_names) == 2 and dv_prob is not None:
        positive_index = list(le.classes_).index("1") if "1" in le.classes_ else 1
        metrics_bin = plot_roc_pr(y_dv, dv_prob[:, positive_index], exp_dir)

    # Save a tiny manifest
    manifest = {
        "dataset": args.dataset_name,
        "features": {"extras": extras, **info_tr},
        "classes": class_names,
        "paths": {
            "confusion_matrix": os.path.join(exp_dir, "confusion_matrix.png"),
            "report_txt": os.path.join(exp_dir, "report.txt")
        }
    }
    manifest.update(metrics_bin)
    with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== DEV RESULTS ===")
    print(rep_txt)
    print(f"\nSaved visuals & reports to: {exp_dir}")
