"""Collect every run under saved_model/ into one table, next to the numbers reported in the paper.

For each run folder it reads results_final.json (written after training by validate(..., tag="final")),
falls back to the newest results_epoch_*.json for runs that have not finished, and reads map.csv for the
best validation mAP. The run settings (construction, gradient estimator, lambda, bit, dataset, seed) are
parsed from the folder name.

Usage:
    python summarize_results.py                              # scans ./saved_model
    python summarize_results.py saved_model --extra results/eval.csv --out results/summary
        --extra  CSV written by eval_codes.py (e.g. the original CSQ / GreedyHash baselines)
        --out    writes <out>.csv and <out>.md

Paper reference: Table 1 (ProbHash / HashUQ / HashUQ-Bin and baselines) and Table 2 (ImageNet ablation,
Center-Target). The paper evaluates with 100 MC Dropout samples; runs evaluated with --sample 1 are
marked in the "sample" column, re-run them with --no-train --sample 100 for the final numbers.
"""
import argparse
import csv
import glob
import json
import os
import re

# ---------------------------------------------------------------- paper numbers (16 / 32 / 64 bits)
BITS = (16, 32, 64)
PAPER = {
    # Table 1
    ("Center", "ProbHash"): {"imagenet": (0.606, 0.680, 0.706), "coco": (0.665, 0.731, 0.764), "nuswide_21": (0.790, 0.820, 0.835)},
    ("Center", "HashUQ"): {"imagenet": (0.631, 0.696, 0.718), "coco": (0.688, 0.743, 0.771), "nuswide_21": (0.814, 0.834, 0.843)},
    ("Center", "HashUQ-Bin"): {"imagenet": (0.623, 0.690, 0.714), "coco": (0.681, 0.739, 0.768), "nuswide_21": (0.805, 0.829, 0.839)},
    ("Label", "ProbHash"): {"imagenet": (0.595, 0.645, 0.673), "coco": (0.653, 0.703, 0.726), "nuswide_21": (0.775, 0.800, 0.816)},
    ("Label", "HashUQ"): {"imagenet": (0.623, 0.655, 0.678), "coco": (0.669, 0.712, 0.730), "nuswide_21": (0.799, 0.812, 0.822)},
    ("Label", "HashUQ-Bin"): {"imagenet": (0.618, 0.653, 0.675), "coco": (0.663, 0.709, 0.729), "nuswide_21": (0.790, 0.806, 0.818)},
    ("Baseline", "CSQ"): {"imagenet": (0.595, 0.668, 0.699), "coco": (0.651, 0.732, 0.769), "nuswide_21": (0.789, 0.820, 0.834)},
    ("Baseline", "GreedyHash"): {"imagenet": (0.611, 0.657, 0.680), "coco": (0.654, 0.702, 0.714), "nuswide_21": (0.776, 0.796, 0.812)},
}
# Table 2: ImageNet, Center-Target, (grad_est, lambda) -> Hamming-ranking mAP
ABLATION = {("st", 0.0): (0.573, 0.660, 0.689), ("st", 1.0): (0.592, 0.666, 0.693), ("cf", 1.0): (0.604, 0.679, 0.707)}

CSQ_RE = re.compile(r"^BerVAE_bit_(?P<bit>\d+)_optimizer_(?P<grad>[a-z0-9]+)_KL_(?P<kl>[\d.]+)_dataset_(?P<dataset>.+?)"
                    r"(?:_net_(?P<net>[^_]+))?(?:_seed_(?P<seed>\d+))?$")
GH_RE = re.compile(r"^BerVAE_Greedy_bit_(?P<bit>\d+)_dataset_(?P<dataset>.+?)_opt_(?P<grad>[a-z0-9]+)_u2gsample_\d+"
                   r"_KL_(?P<kl>[\d.]+)(?:_net_(?P<net>[^_]+))?(?:_seed_(?P<seed>\d+))?$")


def parse_run_name(name):
    """Folder name -> dict(construction, grad, kl, bit, dataset, net, seed), or None if not a HashUQ run."""
    for construction, regex in (("Center", CSQ_RE), ("Label", GH_RE)):
        m = regex.match(name)
        if m:
            d = m.groupdict()
            return {"construction": construction, "grad": d["grad"], "kl": float(d["kl"]), "bit": int(d["bit"]),
                    "dataset": d["dataset"], "net": d["net"] or "alexnet", "seed": int(d["seed"] or 0)}
    return None


def short_label(info):
    lam = ("%g" % info["kl"])
    return "%s %s λ=%s %db %s s%d" % (info["construction"], info["grad"], lam, info["bit"], info["dataset"], info["seed"])


def paper_value(construction, method, dataset, bit):
    try:
        return PAPER[(construction, method)][dataset][BITS.index(bit)]
    except (KeyError, ValueError):
        return None


def paper_hamming(info):
    """Reference for the plain Hamming-ranking mAP of a run (ablation table first, then Table 1)."""
    if info["net"] != "alexnet":
        return None, ""
    if info["construction"] == "Center" and info["dataset"] == "imagenet" and (info["grad"], info["kl"]) in ABLATION \
            and info["bit"] in BITS:
        return ABLATION[(info["grad"], info["kl"])][BITS.index(info["bit"])], "Tab.2"
    is_paper_setting = (info["construction"] == "Center" and info["grad"] == "cf" and info["kl"] == 1.0) or \
                       (info["construction"] == "Label" and info["grad"] == "st" and info["kl"] == 1.0)
    if is_paper_setting:
        v = paper_value(info["construction"], "ProbHash", info["dataset"], info["bit"])
        return v, ("Tab.1" if v is not None else "")
    return None, ""


def read_map_csv(run_dir):
    path = os.path.join(run_dir, "map.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [float(r["0"]) for r in csv.DictReader(f)]


def collect_runs(roots, test_map):
    rows = []
    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
            info = parse_run_name(os.path.basename(run_dir))
            if info is None or not os.path.isdir(run_dir):
                continue
            final = os.path.join(run_dir, "results_final.json")
            status = "done"
            if not os.path.isfile(final):
                partial = sorted(glob.glob(os.path.join(run_dir, "results_epoch_*.json")),
                                 key=lambda p: int(re.findall(r"(\d+)\.json$", p)[0]))
                final = partial[-1] if partial else None
                status = "running/partial"
            res = json.load(open(final)) if final else {}
            maps = read_map_csv(run_dir)
            ref, ref_src = paper_hamming(info)
            row = {"run": os.path.basename(run_dir), "label": short_label(info), **info,
                   "status": status if final else "no results yet", "sample": res.get("sample", ""),
                   "best_val_mAP": round(max(maps), 4) if maps else "",
                   "best_epoch": (maps.index(max(maps)) + 1) * test_map if maps else "",
                   "mAP": round(res["mAP_Hamming"], 4) if "mAP_Hamming" in res else "",
                   "paper_mAP": ref if ref is not None else "", "paper_src": ref_src}
            row["diff"] = round(row["mAP"] - ref, 4) if row["mAP"] != "" and ref is not None else ""
            for key, method in (("mAP_HashUQ", "HashUQ"), ("mAP_HashUQ-Bin_L2", "HashUQ-Bin")):
                row[key] = round(res[key], 4) if key in res else ""
                p = paper_value(info["construction"], method, info["dataset"], info["bit"]) if info["net"] == "alexnet" else None
                row["paper_" + method] = p if (p is not None and key in res) else ""
            rows.append(row)
    return rows


def collect_extra(path):
    """Rows written by eval_codes.py (e.g. original baselines)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("name", "")
            if parse_run_name(name):  # a ProbHash/HashUQ run, already listed from saved_model/
                continue
            method = "CSQ" if "csq" in name.lower() else ("GreedyHash" if "greedy" in name.lower() else None)
            bit = int(r["bit"]) if r.get("bit") else None
            ref = paper_value("Baseline", method, r.get("dataset"), bit) if method else None
            mAP = float(r["mAP"]) if r.get("mAP") else ""
            rows.append({"run": name, "label": name, "construction": "Baseline", "grad": "", "kl": "", "bit": bit,
                         "dataset": r.get("dataset", ""), "seed": "", "status": "eval_codes", "sample": "",
                         "best_val_mAP": "", "best_epoch": "", "mAP": mAP,
                         "paper_mAP": ref if ref is not None else "", "paper_src": "Tab.1" if ref is not None else "",
                         "diff": round(mAP - ref, 4) if (mAP != "" and ref is not None) else "",
                         "mAP_HashUQ": r.get("mAP_HashUQ", ""), "paper_HashUQ": "",
                         "mAP_HashUQ-Bin_L2": r.get("mAP_HashUQ-Bin_L2", ""), "paper_HashUQ-Bin": ""})
    return rows


COLUMNS = ["label", "status", "sample", "best_val_mAP", "best_epoch", "mAP", "paper_mAP", "paper_src", "diff",
           "mAP_HashUQ", "paper_HashUQ", "mAP_HashUQ-Bin_L2", "paper_HashUQ-Bin"]


def to_markdown(rows):
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", default=["saved_model"])
    parser.add_argument("--extra", nargs="*", default=[], help="CSV files from eval_codes.py")
    parser.add_argument("--test_map", type=int, default=10, help="evaluation interval used during training")
    parser.add_argument("--out", default=None, help="write <out>.csv and <out>.md")
    args = parser.parse_args()

    rows = collect_runs(args.roots, args.test_map)
    for path in args.extra:
        rows += collect_extra(path)
    if not rows:
        print("No runs found under", ", ".join(args.roots))
        return
    rows.sort(key=lambda r: (str(r["dataset"]), r["bit"] or 0, r["construction"], str(r["grad"]), str(r["kl"]), str(r["seed"])))

    md = to_markdown(rows)
    print(md)
    if any(r.get("sample") == 1 for r in rows):
        print("\nNote: sample=1 rows were evaluated without MC Dropout; the paper uses 100 samples.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        fields = list(rows[0].keys())
        with open(args.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print("\nWritten %s.csv and %s.md" % (args.out, args.out))


if __name__ == "__main__":
    main()
