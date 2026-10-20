"""Convergence curves for ProbHash runs: training loss, KL term and validation mAP against epoch.

Reads, from each run folder under saved_model/:
    epoch_log.csv   per-epoch training loss and mean KL (written by CSQ_HashUQ.py / GreedyHash_HashUQ.py)
    map.csv         mAP at every evaluation (every --test_map epochs; validation mAP when run with --val)
Older runs without epoch_log.csv fall back to train_loss.csv / kl.csv (one point per evaluation).

Center-Target and Label-Target losses live on different scales (expected Hamming distance vs
cross-entropy), so each construction gets its own row of panels.

Usage:
    python plot_curves.py                                   # every run under ./saved_model -> curves.png
    python plot_curves.py saved_model/BerVAE_bit_16_optimizer_cf_* saved_model/BerVAE_bit_16_optimizer_st_* --out results/ablation_16.png
    python plot_curves.py --filter dataset_imagenet --out results/imagenet.png
"""
import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_results import parse_run_name, short_label

# categorical slots in fixed order + a line style per slot, so series stay distinguishable without color
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def read_columns(path, columns):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {c: [float(r[c]) for r in rows if r.get(c, "") != ""] for c in columns}


def load_run(run_dir, test_map):
    log = read_columns(os.path.join(run_dir, "epoch_log.csv"), ["epoch", "train_loss", "kl"])
    maps = read_columns(os.path.join(run_dir, "map.csv"), ["0"])
    run = {"maps": maps["0"] if maps else []}
    run["map_epochs"] = [(i + 1) * test_map for i in range(len(run["maps"]))]
    if log:
        run["epochs"], run["loss"], run["kl"] = log["epoch"], log["train_loss"], log["kl"]
    else:  # older runs: one point per evaluation
        tl = read_columns(os.path.join(run_dir, "train_loss.csv"), ["0"])
        kl = read_columns(os.path.join(run_dir, "kl.csv"), ["0"])
        run["loss"] = tl["0"] if tl else []
        run["kl"] = kl["0"] if kl else []
        run["epochs"] = [(i + 1) * test_map for i in range(max(len(run["loss"]), len(run["kl"])))]
    return run


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("epoch", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["saved_model"], help="run folders or a folder of runs")
    parser.add_argument("--filter", default="", help="only runs whose folder name contains this text")
    parser.add_argument("--test_map", type=int, default=10, help="evaluation interval used during training")
    parser.add_argument("--out", default="curves.png")
    args = parser.parse_args()

    run_dirs = []
    for p in args.paths:
        cand = [p] if parse_run_name(os.path.basename(os.path.normpath(p))) else sorted(glob.glob(os.path.join(p, "*")))
        run_dirs += [d for d in cand if os.path.isdir(d) and parse_run_name(os.path.basename(os.path.normpath(d)))
                     and args.filter in os.path.basename(os.path.normpath(d))]
    groups = {}
    for d in run_dirs:
        info = parse_run_name(os.path.basename(os.path.normpath(d)))
        run = load_run(d, args.test_map)
        if run["loss"] or run["maps"]:
            groups.setdefault(info["construction"], []).append((short_label(info), run))
    if not groups:
        print("No runs with epoch_log.csv / map.csv found")
        return

    names = sorted(groups)
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 4.2 * len(names)), squeeze=False)
    for row, construction in enumerate(names):
        runs = groups[construction]
        if len(runs) > len(COLORS):
            print("Warning: %d %s runs, only the first %d are drawn - narrow with --filter" % (len(runs), construction, len(COLORS)))
            runs = runs[:len(COLORS)]
        ax_loss, ax_kl, ax_map = axes[row]
        for i, (label, run) in enumerate(runs):
            kw = dict(color=COLORS[i], linestyle=STYLES[i], linewidth=2, label=label)
            if run["loss"]:
                ax_loss.plot(run["epochs"][:len(run["loss"])], run["loss"], **kw)
            if run["kl"]:
                ax_kl.plot(run["epochs"][:len(run["kl"])], run["kl"], **kw)
            if run["maps"]:
                ax_map.plot(run["map_epochs"], run["maps"], marker="o", markersize=4, **kw)
                best = max(range(len(run["maps"])), key=run["maps"].__getitem__)
                ax_map.plot(run["map_epochs"][best], run["maps"][best], marker="*", markersize=13, color=COLORS[i],
                            markeredgecolor="white", markeredgewidth=1, linestyle="none")
                ax_map.annotate("%.3f" % run["maps"][best], (run["map_epochs"][best], run["maps"][best]),
                                textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
                print("%-45s best mAP %.4f at epoch %d" % (label, run["maps"][best], run["map_epochs"][best]))
        loss_name = "expected Hamming loss" if construction == "Center" else "cross-entropy"
        style_axis(ax_loss, "%s-Target: training loss" % construction, loss_name)
        style_axis(ax_kl, "%s-Target: KL to Bernoulli(0.5) prior" % construction, "KL per bit")
        style_axis(ax_map, "%s-Target: mAP at evaluation (★ = best)" % construction, "mAP")
        ax_map.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
