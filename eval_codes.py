"""Shared evaluation of saved hash codes (ProbHash/HashUQ runs and original baselines alike).

Reads the .npy files written by validate() in utils/tools.py:
    tst_binary.npy, trn_binary.npy, tst_label.npy, trn_label.npy  (+ trn_uct.npy for --rt_st hamuct runs)
and reports, on exactly the same query/database codes:
    mAP             mAP@topK, same computation as CalcTopMap (mAP@1000 ImageNet, mAP@5000 COCO / NUS-WIDE)
    P@H<=r          precision of the items within Hamming radius r (0 when nothing falls inside the radius)
    mAP_HashUQ      ranking by Hamming distance + normalized uncertainty, as CalcTopMap_Uct (if trn_uct.npy exists)
    mAP_HashUQ-Bin_Lx  same with the uncertainty quantized into x quantile bins

Exactness. With 16-bit codes almost every database item ties with thousands of others on Hamming distance,
so mAP@topK depends on how np.argsort orders the ties. This script therefore builds the distance vector of
each query exactly like CalcHammingDist (same expression, same dtype as the saved codes, a fresh 1-D array)
and sorts it with the same np.argsort call, so it reproduces the training script's mAP to the last digit.
Only the relevance test (shared label) is vectorized: it is computed with float32 matrix products, which are
exact for 0/1 labels and ~30x faster than the int64 products of the original loop.

Only numpy is needed, so it also runs on a laptop without a GPU.

Usage:
    python eval_codes.py saved_model/                      # every run under saved_model
    python eval_codes.py saved_model/BerVAE_bit_16_optimizer_cf_KL_1.00_dataset_imagenet_net_alexnet_seed_0
    python eval_codes.py ../baseline/save/CSQ --name CSQ-baseline --radius 2 --out results/eval.csv

A path may be a folder that directly holds the .npy files, a run folder, or any parent of run folders.
When a run folder holds several code folders, the most recently written one is used (the final
evaluation of the best checkpoint is always written last).
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np

EVAL_CODES_VERSION = "2"   # 2: exact CalcHammingDist mirror (dtype/tie order), float32 relevance, progress output
TOPK = {"imagenet": 1000, "coco": 5000, "nuswide_21": 5000, "nuswide_21_m": 5000, "nuswide_81_m": 5000}
CODE_FILES = ["tst_binary.npy", "trn_binary.npy", "tst_label.npy", "trn_label.npy"]


# ---------------------------------------------------------------- metrics (kept identical to utils/tools.py)
def hamming(q_code, rB):
    """CalcHammingDist(B1, B2) for one query: 0.5 * (q - B1 . B2^T), in the dtype of the codes."""
    return 0.5 * (rB.shape[1] - np.dot(q_code, rB.transpose()))


def ap_at_k(gnd_sorted, topk):
    tgnd = gnd_sorted[:topk]
    tsum = np.sum(tgnd).astype(int)
    if tsum == 0:
        return 0.0
    count = np.linspace(1, tsum, tsum)
    tindex = np.asarray(np.where(tgnd == 1)) + 1.0
    return float(np.mean(count / tindex))


def find_quantile(values, quant_level):
    # utils.tools.find_quantile(..., equal=False)
    return [np.quantile(values.reshape(-1), i / quant_level) for i in range(quant_level + 1)]


def quantize_data(values, quantile):
    return np.array(quantile)[np.digitize(values, quantile, right=True)]


def evaluate(qB, rB, qL, rL, topk, radius=None, r_uct=None, levels=(), chunk=512, progress=True):
    """Returns a dict of metrics, numerically identical to CalcTopMap / CalcTopMap_Uct."""
    # codes are used as saved (float32 from validate()); integer codes are promoted like np.dot would need
    if not np.issubdtype(qB.dtype, np.floating):
        qB = qB.astype(np.float32)
    if not np.issubdtype(rB.dtype, np.floating):
        rB = rB.astype(np.float32)
    if topk is None or topk < 0:
        topk = rB.shape[0]
    rankers = {"mAP": None}
    if r_uct is not None:
        r_uct = np.asarray(r_uct).reshape(-1)
        if not np.all(np.isfinite(r_uct)):
            print("  WARNING: trn_uct contains NaN/inf (%d values) - HashUQ numbers are not trustworthy"
                  % int(np.sum(~np.isfinite(r_uct))))
        rankers["mAP_HashUQ"] = r_uct
        for level in levels:
            rankers["mAP_HashUQ-Bin_L%d" % level] = quantize_data(r_uct, find_quantile(r_uct, level))

    qLf = np.asarray(qL, dtype=np.float32)
    rLTf = np.ascontiguousarray(np.asarray(rL, dtype=np.float32).T)
    sums = {k: 0.0 for k in rankers}
    prec_sum = 0.0
    n_query = qB.shape[0]
    t0, step = time.time(), max(1, n_query // 10)
    for start in range(0, n_query, chunk):
        gnd_block = (qLf[start:start + chunk] @ rLTf > 0).astype(np.float32)   # == (qL[i] . rL^T > 0)
        for j, gnd in enumerate(gnd_block):
            i = start + j
            hamm = hamming(qB[i, :], rB)
            for name, offset in rankers.items():
                ind = np.argsort(hamm if offset is None else hamm + offset)
                sums[name] += ap_at_k(gnd[ind], topk)
            if radius is not None:
                inside = hamm <= radius + 1e-6
                n_inside = int(inside.sum())
                prec_sum += float(gnd[inside].sum()) / n_inside if n_inside > 0 else 0.0
            if progress and ((i + 1) % step == 0 or i + 1 == n_query):
                el = time.time() - t0
                print("    %6d/%d queries  %5.0fs  (~%.0fs left)" % (i + 1, n_query, el, el / (i + 1) * (n_query - i - 1)),
                      flush=True)
    out = {k: v / n_query for k, v in sums.items()}
    if radius is not None:
        out["P@H<=%d" % radius] = prec_sum / n_query
    return out


# ---------------------------------------------------------------- locating saved codes
def has_codes(folder):
    return all(os.path.isfile(os.path.join(folder, f)) for f in CODE_FILES)


def find_code_dirs(path):
    """Map run folder -> newest code folder below it."""
    path = os.path.normpath(path)
    if has_codes(path):
        return {path: path}
    found = {}
    for root, _, files in os.walk(path):
        if "tst_binary.npy" in files and has_codes(root):
            run = os.path.dirname(root)
            if run not in found or os.path.getmtime(os.path.join(root, "tst_binary.npy")) > \
                    os.path.getmtime(os.path.join(found[run], "tst_binary.npy")):
                found[run] = root
    return dict(sorted(found.items()))


def describe(code_dir):
    """dataset / bit from the folder name '<dataset>_<bit>bits_<mAP>' written by validate()."""
    m = re.match(r"^(.+)_(\d+)bits_", os.path.basename(code_dir))
    return (m.group(1), int(m.group(2))) if m else (None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--dataset", default=None, help="imagenet / coco / nuswide_21 (otherwise read from folder name)")
    parser.add_argument("--topk", type=int, default=None, help="override topK (default: 1000 ImageNet, 5000 COCO/NUS-WIDE)")
    parser.add_argument("--radius", type=int, default=2, help="Hamming radius for P@H<=r, -1 to skip")
    parser.add_argument("--levels", type=int, nargs="*", default=[2, 4], help="HashUQ-Bin quantization levels")
    parser.add_argument("--no-uct", dest="uct", action="store_false", help="ignore trn_uct.npy even if present")
    parser.add_argument("--name", default=None, help="label for the result rows (default: run folder name)")
    parser.add_argument("--out", default=None, help="append results to this CSV file")
    parser.add_argument("--quiet", action="store_true", help="no per-query progress lines")
    args = parser.parse_args()

    rows = []
    for path in args.paths:
        code_dirs = find_code_dirs(path)
        if not code_dirs:
            print("No saved codes (%s) found under %s" % (", ".join(CODE_FILES), path))
            continue
        for run, code_dir in code_dirs.items():
            dataset, bit = describe(code_dir)
            dataset = args.dataset or dataset
            topk = args.topk if args.topk is not None else TOPK.get(dataset)
            if topk is None:
                sys.exit("Cannot tell topK for %s: pass --dataset or --topk" % code_dir)
            qB, rB, qL, rL = [np.load(os.path.join(code_dir, f)) for f in CODE_FILES]
            uct_file = os.path.join(code_dir, "trn_uct.npy")
            r_uct = np.load(uct_file) if (args.uct and os.path.isfile(uct_file)) else None
            print("Evaluating %s (%d queries x %d database, %d bits, top%d)" % (code_dir, qB.shape[0], rB.shape[0], qB.shape[1], topk),
                  flush=True)
            res = evaluate(qB, rB, qL, rL, topk, radius=None if args.radius < 0 else args.radius,
                           r_uct=r_uct, levels=args.levels, progress=not args.quiet)
            row = {"name": args.name or os.path.basename(run), "dataset": dataset, "bit": bit or qB.shape[1],
                   "topK": topk, "code_dir": code_dir}
            row.update({k: float(v) for k, v in res.items()})   # full precision, so it can be compared exactly
            rows.append(row)
            print("  " + "  ".join("%s=%.4f" % (k, v) for k, v in res.items()), flush=True)

    if args.out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        old_rows, fields = [], []
        if os.path.isfile(args.out):  # keep earlier rows; widen the header if new metrics appear
            with open(args.out, newline="") as f:
                reader = csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                old_rows = list(reader)
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(old_rows + rows)
        print("Results appended to", args.out)


if __name__ == "__main__":
    main()
