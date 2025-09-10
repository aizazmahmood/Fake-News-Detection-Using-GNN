import argparse, json, torch
from ast import literal_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--drop", nargs="*", default=[], help="Edge type tuples as strings, e.g. \"('article','similar','article')\"")
    args = ap.parse_args()

    data = torch.load(args.in_pt)
    drops = [literal_eval(s) for s in args.drop]
    for et in drops:
        if et in data.edge_types:
            del data[et]
            print("Dropped", et)
        # also drop reverse if present
        rev = (et[2], f"rev_{et[1]}", et[0])
        if rev in data.edge_types:
            del data[rev]
            print("Dropped", rev)

    torch.save(data, args.out_pt)
    # copy the same json meta (edge list differences don’t affect label map)
    j_in, j_out = args.in_pt.replace(".pt",".json"), args.out_pt.replace(".pt",".json")
    try:
        meta = json.load(open(j_in))
        json.dump(meta, open(j_out,"w"), indent=2)
    except FileNotFoundError:
        pass
    print("Saved →", args.out_pt)

if __name__ == "__main__":
    main()
