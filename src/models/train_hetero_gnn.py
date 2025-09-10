import os, argparse, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def focal_loss(logits, targets, weight=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata, raw_in_dims, hidden=64, out_classes=4, drop=0.3, aggr='sum'):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types

        base_dim = max(int(raw_in_dims[nt]) for nt in node_types)
        self.enc = torch.nn.ModuleDict({nt: torch.nn.Linear(int(raw_in_dims[nt]), base_dim) for nt in node_types})

        def layer(out_dim):
            return HeteroConv({et: SAGEConv((-1, -1), out_dim) for et in edge_types}, aggr=aggr)

        self.conv1 = layer(hidden)
        self.conv2 = layer(hidden)
        self.drop  = torch.nn.Dropout(drop)
        self.out   = torch.nn.Linear(hidden, out_classes)

    def forward(self, x_dict, edge_index_dict):
        x = {nt: F.relu(self.enc[nt](x_dict[nt].float())) for nt in x_dict}
        x = self.conv1(x, edge_index_dict)
        x = {k: self.drop(F.relu(v)) for k,v in x.items()}
        x = self.conv2(x, edge_index_dict)
        x = {k: self.drop(F.relu(v)) for k,v in x.items()}
        return self.out(x["article"])

def plot_confusion(cm, labels, out_png):
    fig = plt.figure(figsize=(6,5))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = plt.imshow(cmn); plt.title("Confusion Matrix"); plt.colorbar(im)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right"); plt.yticks(ticks, labels)
    thr = cmn.max()/2
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            v = cmn[i,j]
            plt.text(j,i,f"{v:.2f}",ha="center",va="center",
                     color="white" if v>thr else "black", fontsize=8)
    plt.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

def evaluate(model, data, mask, id2label, out_png=None):
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        pred = logits[mask].argmax(-1).cpu().numpy()
        true = data["article"].y[mask].cpu().numpy()
    rep_txt = classification_report(true, pred,
                                   target_names=[id2label[i] for i in range(len(id2label))],
                                   digits=4, zero_division=0)
    cm = confusion_matrix(true, pred, labels=list(range(len(id2label))))
    if out_png:
        plot_confusion(cm, [id2label[i] for i in range(len(id2label))], out_png)
    return rep_txt, cm, f1_score(true, pred, average="macro")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.out_dir)

    data: HeteroData = torch.load(args.graph_path)
    meta = json.load(open(args.graph_path.replace(".pt",".json")))
    id2label = {i:l for l,i in meta["label_map"].items()}
    raw_in_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}

    for nt in data.node_types:
        data[nt].x = data[nt].x.to(device, dtype=torch.float32)
    data["article"].y = data["article"].y.to(device)
    for et in data.edge_types:
        data[et].edge_index = data[et].edge_index.to(device)

    model = HeteroSAGE(data.metadata(), raw_in_dims,
                       hidden=args.hidden, out_classes=meta["num_classes"],
                       drop=args.dropout, aggr=args.aggr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # class weights (tunable power to avoid over-penalizing)
    cls = data["article"].y[data["article"].train_mask]
    cnt = torch.bincount(cls, minlength=meta["num_classes"]).float()
    inv = (cnt.sum() / torch.clamp(cnt, min=1.0))
    cw  = inv.pow(args.cw_pow).to(device)  # cw_pow=1.0 -> inverse freq; 0.5 -> sqrt; 0 -> none

    best_state, best_f1, wait = None, -1.0, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        opt.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        idx = data["article"].train_mask
        if args.focal_gamma > 0:
            loss = focal_loss(logits[idx], data["article"].y[idx], weight=cw, gamma=args.focal_gamma)
        else:
            loss = F.cross_entropy(logits[idx], data["article"].y[idx], weight=cw)
        loss.backward(); opt.step()

        rep_txt, _, macro_f1 = evaluate(model, data, data["article"].val_mask, id2label)
        print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | dev macro-F1 {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1, wait, best_state = macro_f1, 0, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stop."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    dev_rep, dev_cm, _ = evaluate(model, data, data["article"].val_mask,  id2label,
                                  out_png=os.path.join(args.out_dir, "dev_confusion.png"))
    test_rep, test_cm, _ = evaluate(model, data, data["article"].test_mask, id2label,
                                    out_png=os.path.join(args.out_dir, "test_confusion.png"))

    open(os.path.join(args.out_dir, "dev_report.txt"), "w", encoding="utf-8").write(dev_rep)
    open(os.path.join(args.out_dir, "test_report.txt"), "w", encoding="utf-8").write(test_rep)
    np.savetxt(os.path.join(args.out_dir, "dev_confusion.csv"), dev_cm, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(args.out_dir, "test_confusion.csv"), test_cm, fmt="%d", delimiter=",")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "heterosage_best.pt"))
    print("\n=== DEV ===\n", dev_rep)
    print("\n=== TEST ===\n", test_rep)
    print(f"\nSaved model → {os.path.join(args.out_dir, 'heterosage_best.pt')}")
    print(f"Plots saved → {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_path", required=False, default="data/processed/graphs/buzzfeed_no_sim.pt")
    ap.add_argument("--out_dir",    required=False, default="outputs/gnn/bf_hetero_sage_no_sim_focal")
    ap.add_argument("--hidden",     type=int, default=64)
    ap.add_argument("--dropout",    type=float, default=0.3)
    ap.add_argument("--lr",         type=float, default=2e-3)
    ap.add_argument("--epochs",     type=int, default=200)
    ap.add_argument("--patience",   type=int, default=20)
    ap.add_argument("--aggr",       default="sum", choices=["sum","mean","max"])
    ap.add_argument("--focal_gamma",type=float, default=2.0)   # 0 disables focal
    ap.add_argument("--cw_pow",     type=float, default=0.5)   # 0=no weights, 1=inverse freq
    ap.add_argument("--cpu",        action="store_true")
    args = ap.parse_args()
    main(args)
