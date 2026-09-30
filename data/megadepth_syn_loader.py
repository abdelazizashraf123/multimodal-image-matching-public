#!/usr/bin/env python3
"""Load MegaDepth-Syn (MINIMA) from data/megadepth_syn: synthetic modalities of MegaDepth."""
import argparse
import glob
import os.path as osp

DEFAULT_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "megadepth_syn")
MEGA_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "megadepth")


def modalities(root=DEFAULT_ROOT, split="train"):
    """Modality names present on disk, e.g. Infrared, Depth."""
    return sorted(osp.basename(p) for p in glob.glob(osp.join(root, split, "*"))
                  if osp.isdir(p))


def images(root=DEFAULT_ROOT, modality="Infrared", split="train", scene=None):
    """All image paths for one modality, optionally one scene."""
    pat = osp.join(root, split, modality, "MegaDepth_v1",
                   scene or "*", "dense*", "imgs", "*.jpg")
    return sorted(glob.glob(pat))


def scenes(root=DEFAULT_ROOT, modality="Infrared", split="train"):
    """Scene ids present for one modality."""
    pat = osp.join(root, split, modality, "MegaDepth_v1", "*")
    return sorted(osp.basename(p) for p in glob.glob(pat) if osp.isdir(p))


def pair_with_megadepth(syn_path, mega_root=MEGA_ROOT):
    """RGB counterpart of a synthetic image, or None if it is not on disk.

    Filenames match the real MegaDepth ones, so only the prefix differs.
    """
    tail = syn_path.split("MegaDepth_v1" + osp.sep, 1)
    if len(tail) != 2:
        return None
    rgb = osp.join(mega_root, "phoenix", "S6", "zl548", "MegaDepth_v1", tail[1])
    return rgb if osp.exists(rgb) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--modality", default=None, help="default: summarise all")
    a = ap.parse_args()

    mods = [a.modality] if a.modality else modalities(a.root, a.split)
    if not mods:
        print(f"nothing found in {a.root} -- see data/README.md for the download link")
        return
    for m in mods:
        imgs = images(a.root, m, a.split)
        sc = scenes(a.root, m, a.split)
        paired = sum(1 for p in imgs[:200] if pair_with_megadepth(p))
        print(f"{m:10s} {len(sc):4d} scenes  {len(imgs):7d} images  "
              f"{paired}/200 sampled have an RGB counterpart")


if __name__ == "__main__":
    main()
