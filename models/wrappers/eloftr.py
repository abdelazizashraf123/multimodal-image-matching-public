"""Original EfficientLoFTR (eloftr_outdoor.ckpt), loaded the way its own repository does it.

Two inference configurations of the SAME weights, both from the repository README:
  * mode="full" (default): full_default_cfg, fp32 -- "best quality". This produced the group's
    ELoFTR rows (a-shahzeen's ELoFTRMatcher in extra_matchers.py), so the numbers are comparable.
  * mode="opt": opt_default_cfg (coarse threshold 25, softmax skipped, fp16 matmul in the coarse
    matcher) plus precision 'fp16' (cfg['half'] = True, model.half(), fp16 inputs) -- "best
    efficiency". Tag eloftr_opt. Used for the version comparison (2026-09-16).
Common to both: checkpoint 'matcher.' prefix stripped, then reparameter() -- the README calls this
"essential for good performance" (RepVGG branches folded at inference); grayscale input, each
image resized to the multiple of 32 at or below its own size (the model's requirement), keypoints
scaled back to the input frame.
Code and weights are the user's own clone/download: ~/projects/EfficientLoFTR and
/cta/scratch/h-abdelaziz/weights/ELoFTR/weights/eloftr_outdoor.ckpt (md5 b9f59b9b...).
"""
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch

ELOFTR = "/cta/users/h-abdelaziz/projects/EfficientLoFTR"
CKPT = "/cta/scratch/h-abdelaziz/weights/ELoFTR/weights/eloftr_outdoor.ckpt"


def _shim_create_meshgrid_dtype():
    """The repository pins kornia 0.4.1 (requirements.txt) but its fp16 path calls
    create_meshgrid(..., dtype=torch.float16), an argument that exists only from kornia 0.5.1
    (the authors' own comment on that line). Rather than upgrade kornia under the validated full-mode
    numbers, accept the argument here and cast afterwards: for these small integer grids the values
    are identical to building the grid in fp16 directly."""
    import src.loftr.utils.fine_matching as fm
    orig = fm.create_meshgrid
    if getattr(orig, "_dtype_shim", False):
        return

    def create_meshgrid(height, width, normalized_coordinates=True, device=None, dtype=None):
        g = orig(height, width, normalized_coordinates, device)
        return g.to(dtype) if dtype is not None else g

    create_meshgrid._dtype_shim = True
    fm.create_meshgrid = create_meshgrid


class ELoFTRMatcher:
    def __init__(self, ckpt=CKPT, device="cuda", mode="full"):
        if mode not in ("full", "opt"):
            raise ValueError(f"mode must be 'full' or 'opt', got {mode!r}")
        # EfficientLoFTR's package is called `src`; make sure no other repo's `src` is loaded.
        for m in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[m]
        if ELOFTR not in sys.path:
            sys.path.insert(0, ELOFTR)
        from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
        cfg = deepcopy(opt_default_cfg if mode == "opt" else full_default_cfg)
        self.half = mode == "opt"
        if self.half:
            cfg["half"] = True                       # README: precision = 'fp16'
            _shim_create_meshgrid_dtype()
        self.model = LoFTR(config=cfg)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        self.model.load_state_dict({k.replace("matcher.", "", 1): v for k, v in sd.items()}, strict=True)
        self.model = reparameter(self.model)
        if self.half:
            self.model = self.model.half()
        self.model = self.model.eval().to(device)
        self.device = device
        self.mode = mode

    def _prep(self, gray):
        h, w = gray.shape
        H, W = max(32, (h // 32) * 32), max(32, (w // 32) * 32)
        t = torch.from_numpy(cv2.resize(gray, (W, H)))[None][None].to(self.device)
        t = (t.half() if self.half else t.float()) / 255.0
        return t, np.array([w / W, h / H], dtype=np.float32)

    @torch.no_grad()
    def __call__(self, i0, i1):
        t0, s0 = self._prep(cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY))
        t1, s1 = self._prep(cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY))
        b = {"image0": t0, "image1": t1}
        self.model(b)
        return b["mkpts0_f"].float().cpu().numpy() * s0, b["mkpts1_f"].float().cpu().numpy() * s1


def build_matcher(a, device="cuda", mode="full"):
    ckpt = getattr(a, "extra_ckpt", None) or CKPT
    m = ELoFTRMatcher(ckpt, device, mode)

    def f(i0, i1):          # closure: complexity/bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or ("eloftr_outdoor" if mode == "full" else "eloftr_opt"))
