import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.seed import set_seed

def stratified_split(df, label_col, test_size=0.15, dev_size=0.15, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
    train_df, dev_df  = train_test_split(train_df, test_size=dev_size/(1-test_size), stratify=train_df[label_col], random_state=seed)
    return train_df, dev_df, test_df

def main(in_csv, label_col, out_dir):
    set_seed(42)
    df = pd.read_csv(in_csv)
    os.makedirs(out_dir, exist_ok=True)
    tr, dv, te = stratified_split(df, label_col)
    tr.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    dv.to_csv(os.path.join(out_dir, "dev.csv"), index=False)
    te.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"{len(tr)=}, {len(dv)=}, {len(te)=}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.in_csv, args.label_col, args.out_dir)
