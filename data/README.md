# Data

One folder per dataset. Download into the matching folder, then use the loaders.

| Folder | Dataset | Link |
|---|---|---|
| `megadepth/` | MegaDepth v1 (images + depth) | https://www.cs.cornell.edu/projects/megadepth/ |
| `megadepth/scene_info/` | Pair/pose preprocessing (DKM instructions) | https://github.com/Parskatt/DKM |
| `megadepth_syn/` | MegaDepth-Syn, MINIMA's synthetic modalities | https://huggingface.co/datasets/lsxi77777/MegaDepth-Syn |

## Layout

```
data/
  megadepth/
    phoenix/S6/zl548/MegaDepth_v1/<scene>/dense*/imgs/*.jpg
    scene_info/*.npz
  megadepth_syn/
    train/<modality>/MegaDepth_v1/<scene>/dense*/imgs/*.jpg
    test/...
```

`<modality>` is Infrared, Depth, Event, Normal, Paint, Sketch.

## Loaders

```bash
python data/megadepth_loader.py      --root data/megadepth
python data/megadepth_syn_loader.py  --root data/megadepth_syn --modality Infrared
```

Both print a summary when run directly, and expose functions to import.

## Notes

MegaDepth-Syn filenames match the real MegaDepth images, so a synthetic image pairs with
the RGB one of the same name. `megadepth_syn_loader.pair_with_megadepth` does that lookup.

Downloading MegaDepth-Syn from Hugging Face anonymously is rate limited; a token makes it
much faster.
