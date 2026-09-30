#!/usr/bin/env python3
"""Load MegaDepth from data/megadepth: scene files and the image pairs inside them."""
import argparse
import glob
import os.path as osp

import numpy as np

DEFAULT_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "megadepth")


def scene_files(root=DEFAULT_ROOT):
    """Scene files, in either layout that is around: scene_info/*.npz or scene_npz/*.npy."""
    out = []
    for sub, ext in (("scene_info", "npz"), ("scene_info", "npy"),
                     ("scene_npz", "npy"), ("scene_npz", "npz")):
        out += glob.glob(osp.join(root, sub, f"*.{ext}"))
    return sorted(set(out))


def load_scene(path):
    """One scene file as a dict."""
    d = np.load(path, allow_pickle=True)
    return d.item() if isinstance(d, np.ndarray) else {k: d[k] for k in d.files}


def pairs(scene, root=DEFAULT_ROOT, min_overlap=0.1, max_overlap=0.7):
    """Yield (img0, img1, overlap) for one scene, filtered by overlap."""
    s = load_scene(scene) if isinstance(scene, str) else scene
    paths = s["image_paths"]

    if "pairs" in s:                                  # explicit pair list
        ov = s.get("overlaps", [None] * len(s["pairs"]))
        for (i, j), o in zip(s["pairs"], ov):
            if o is None or min_overlap < o < max_overlap:
                yield _abs(paths[i], root), _abs(paths[j], root), \
                    (float(o) if o is not None else None)
        return

    m = s.get("overlap_matrix")                       # dense overlap matrix
    if m is None:
        return
    for i, j in zip(*np.where((m > min_overlap) & (m < max_overlap))):
        if i < j and paths[i] is not None and paths[j] is not None:
            yield _abs(paths[i], root), _abs(paths[j], root), float(m[i, j])


def _abs(p, root):
    """Paths may be absolute already, or relative to the MegaDepth root."""
    return p if osp.isabs(p) else osp.join(root, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=3, help="scenes to summarise")
    a = ap.parse_args()

    files = scene_files(a.root)
    print(f"{len(files)} scenes in {a.root}")
    if not files:
        print("nothing found -- see data/README.md for the download links")
        return
    for f in files[: a.limit]:
        s = load_scene(f)
        n = sum(1 for _ in pairs(s, a.root))
        print(f"  {osp.basename(f):16s} {len(s['image_paths']):5d} images  {n:6d} pairs")


if __name__ == "__main__":
    main()
