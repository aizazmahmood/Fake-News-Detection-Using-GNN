import os, argparse, json, math, random
import numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)

# ---------------- utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def plot_confusion(cm, labels, out_png, normalize=True, title="Confusion Matrix"):
    fig = plt.figure(figsize=(6,5))
    cmn = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) if normalize else cm
    im = plt.imshow(cmn, interpolation='nearest')
    plt.title(title); plt.colorbar(im)
    tick = np.arange(len(labels)); plt.xticks(tick, labels, rotation=45, ha="right"); plt.yticks(tick, labels)
    fmt = ".2f" if normalize else "d"; thr = cmn.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            t = format(cmn[i,j], fmt)
            plt.text(j,i,t,ha="center",va="center",color="white" if cmn[i,j]>thr else "black", fontsize=8)
    plt.tight_layout(); plt.xlabel("Predicted"); plt.ylabel("True")
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

def plot_pr_roc(y_true, y_score, out_dir):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    fig = plt.figure(figsize=(6,5)); plt.plot(fpr,tpr,label=f"AUC={roc:.4f}"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    fig.savefig(os.path.join(out_dir,"roc_curve.png"), bbox_inches="tight"); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_score); pr_auc = auc(rec,prec)
    fig = plt.figure(figsize=(6,5)); plt.plot(rec,prec,label=f"AP={pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    fig.savefig(os.path.join(out_dir,"pr_curve.png"), bbox_inches="tight"); plt.close(fig)
    return roc, pr_auc

# --------------- dataset ----------------
class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, tokenizer, label2id: Dict[str,int], max_len=256):
        self.texts = df[text_col].fillna("").astype(str).tolist()
        self.labels = [label2id[str(x)] for x in df[label_col].astype(str).tolist()]
        self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": self.labels[idx]}

# ------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--test_csv", required=False)
    ap.add_argument("--text_col", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--model_name", default="microsoft/mpnet-base")   # good for short texts; try bert-base-uncased too
    ap.add_argument("--out_dir", default="outputs/hf")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--class_weight", action="store_true", help="use inverse-freq class weights")
    args = ap.parse_args()

    set_seed(args.seed); ensure_dir(args.out_dir)

    tr = pd.read_csv(args.train_csv); dv = pd.read_csv(args.dev_csv)
    if args.test_csv and os.path.exists(args.test_csv):
        te = pd.read_csv(args.test_csv)
    else:
        te = None

    # labels
    labels_sorted = sorted(tr[args.label_col].astype(str).unique().tolist())
    label2id = {l:i for i,l in enumerate(labels_sorted)}
    id2label = {i:l for l,i in label2id.items()}
    num_labels = len(label2id)

    # tokenizer & datasets
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ds_tr = CSVDataset(tr, args.text_col, args.label_col, tok, label2id, args.max_len)
    ds_dv = CSVDataset(dv, args.text_col, args.label_col, tok, label2id, args.max_len)
    ds_te = CSVDataset(te, args.text_col, args.label_col, tok, label2id, args.max_len) if te is not None else None

    # class weights
    cw = None
    if args.class_weight:
        counts = np.bincount([label2id[str(x)] for x in tr[args.label_col].astype(str)])
        inv = 1.0 / np.clip(counts, 1, None)
        cw = torch.tensor(inv * (len(counts)/inv.sum()), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    if cw is not None:
        model.config.problem_type = "single_label_classification"
        # wrap loss via Trainer compute_loss hook later
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # training args
    targs = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        seed=args.seed,
        logging_steps=50
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        rep = classification_report(labels, preds, output_dict=True, digits=4, zero_division=0)
        # return a few key metrics; full report saved after evaluation
        macro_f1 = rep["macro avg"]["f1-score"]
        acc = rep["accuracy"]
        return {"macro_f1": macro_f1, "accuracy": acc}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw.to(logits.device) if cw is not None else None)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_dv,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    trainer.train()

    # ---------- Evaluation & visuals on dev (and test if provided) ----------
    def full_eval(ds, df, split_name):
        out = trainer.predict(ds)
        preds = out.predictions.argmax(-1)
        rep = classification_report(df[args.label_col].map(label2id), preds, target_names=[id2label[i] for i in range(num_labels)], digits=4, output_dict=True, zero_division=0)
        rep_txt = classification_report(df[args.label_col].map(label2id), preds, target_names=[id2label[i] for i in range(num_labels)], digits=4, zero_division=0)
        with open(os.path.join(args.out_dir, f"{split_name}_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep_txt)

        cm = confusion_matrix(df[args.label_col].map(label2id), preds, labels=list(range(num_labels)))
        plot_confusion(cm, [id2label[i] for i in range(num_labels)], os.path.join(args.out_dir, f"{split_name}_confusion.png"))

        # Binary curves
        if num_labels == 2:
            probs = torch.softmax(torch.tensor(out.predictions), dim=1).numpy()[:,1]
            roc, pr = plot_pr_roc(df[args.label_col].map(label2id).values, probs, args.out_dir)
            with open(os.path.join(args.out_dir, f"{split_name}_curves.json"), "w") as f:
                json.dump({"roc_auc": float(roc), "pr_auc": float(pr)}, f, indent=2)

        return rep

    dev_rep = full_eval(ds_dv, dv, "dev")
    if ds_te is not None:
        test_rep = full_eval(ds_te, te, "test")

    print("\n=== DEV (HF) ===")
    print(json.dumps(dev_rep, indent=2))
    print(f"\nSaved outputs → {args.out_dir}")

if __name__ == "__main__":
    main()
