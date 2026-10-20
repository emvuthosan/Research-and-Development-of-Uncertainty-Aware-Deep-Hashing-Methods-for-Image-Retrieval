"""Check an ImageNet-100 download against the split lists in ./data/imagenet and (optionally) rewrite the lists
so that every line points to where the file really is.

The lists in the repo refer to original ILSVRC file names:
    train / database : image/n03944341_10000.JPEG            (100 classes, 128,503 images)
    test (query)     : val_image/ILSVRC2012_val_00000007.JPEG (5,000 validation images)
This script only looks at the file *name*, so it works for any folder layout (class sub-folders, flat, nested).

Usage (from the repo root):
    python scripts/prepare_imagenet_lists.py --root "E:\\Đồ Án Tốt Nghiệp\\Data\\imagenet"            # check only
    python scripts/prepare_imagenet_lists.py --root "E:\\Đồ Án Tốt Nghiệp\\Data\\imagenet" --write    # rewrite lists

--write puts train.txt, new_train.txt, val.txt, test.txt, database.txt (and pool.txt) INSIDE --root, with paths
relative to --root. It refuses to write while any file is missing. Afterwards make ../image_hashing_data/imagenet
point to --root (Windows, no admin needed):
    mklink /J ..\\image_hashing_data\\imagenet "E:\\Đồ Án Tốt Nghiệp\\Data\\imagenet"
"""
import argparse
import os
import re
import sys

LISTS = ["train", "new_train", "val", "test", "database", "pool"]
IMG_EXT = (".jpeg", ".jpg", ".png")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder that holds the downloaded images (any depth)")
    ap.add_argument("--lists", default=os.path.join("data", "imagenet"), help="folder with the repo's list files")
    ap.add_argument("--write", action="store_true", help="write rewritten lists (only if nothing is missing)")
    ap.add_argument("--out", default=None, help="folder for the rewritten lists (default: --root; use it when --root is read-only, e.g. /kaggle/input)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    index, dup, total = {}, 0, 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                total += 1
                rel = os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/")
                if f in index:
                    dup += 1
                else:
                    index[f] = rel
    print("Images found under %s: %d (%d distinct names, %d duplicated names)" % (root, total, len(index), dup))
    if not index:
        sys.exit("No images found - check --root (the folder must contain the .JPEG files somewhere below it)")

    lower_index = {k.lower(): v for k, v in index.items()}
    wnids = set()
    results, ok = {}, True
    for name in LISTS:
        path = os.path.join(args.lists, name + ".txt")
        if not os.path.isfile(path):
            continue
        rows, missing = [], []
        for line in open(path, encoding="utf-8").read().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            base = os.path.basename(parts[0])
            if name == "database":
                m = re.match(r"(n\d{8})_", base)
                if m:
                    wnids.add(m.group(1))
            rel = index.get(base) or lower_index.get(base.lower())
            if rel is None:
                missing.append(parts[0])
            else:
                rows.append(rel + " " + " ".join(parts[1:]))
        results[name] = rows
        print("%-10s %7d lines  found %7d  missing %7d" % (name + ".txt", len(rows) + len(missing), len(rows), len(missing)))
        if missing:
            ok = False
            print("           e.g. " + ", ".join(missing[:3]))

    present = {w for w in wnids if any(k.startswith(w + "_") for k in index)} if len(index) < 3_000_000 else set()
    print("Classes of the 100-class list that have at least one image here: %d / %d" % (len(present), len(wnids)))

    if not ok:
        print("\nNOT READY: some listed files are missing. This download is not the same 100-class ImageNet subset "
              "(or the validation images keep different names). Paste this output to the group before training.")
        sys.exit(1)
    print("\nAll listed files found.")
    if args.write:
        out = os.path.abspath(args.out) if args.out else root
        os.makedirs(out, exist_ok=True)
        for name, rows in results.items():
            with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(rows) + "\n")
        print("Lists written into", out)
        print('Next: mklink /J ..\\image_hashing_data\\imagenet "%s"   then   python check_env.py --dataset imagenet' % root)
    else:
        print("Run again with --write to rewrite the lists.")


if __name__ == "__main__":
    main()
