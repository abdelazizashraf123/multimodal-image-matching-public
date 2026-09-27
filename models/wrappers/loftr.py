"""Original LoFTR (CVPR 2021, github.com/zju3dv/LoFTR), the predecessor of EfficientLoFTR: the
"old version" in the ELoFTR version comparison. outdoor_ds.ckpt (dual-softmax, MegaDepth) with the
repository's `default_cfg`, loaded the way its README does it.

Input handling copies matchers/eloftr.py so the two versions see identical tensors: grayscale,
each image resized to the multiple of 32 at or below its own size (LoFTR itself only needs a
multiple of 8), keypoints scaled back to the input frame, fp32. Code and weights are the user's
own clone/download: ~/projects/LoFTR and /cta/scratch/h-abdelaziz/weights/LoFTR/weights/outdoor_ds.ckpt.
Runs in ~/eloftr_env (LoFTR pins the same kornia 0.4.1 / einops 0.3.0 as ELoFTR).
"""
import sys

import cv2
import numpy as np
import torch

LOFTR = "/cta/users/h-abdelaziz/projects/LoFTR"
CKPT = "/cta/scratch/h-abdelaziz/weights/LoFTR/weights/outdoor_ds.ckpt"


class LoFTRMatcher:
    def __init__(self, ckpt=CKPT, device="cuda"):
        # LoFTR's package is also called `src`; make sure no other repo's `src` is loaded.
        for m in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[m]
        if LOFTR not in sys.path:
            sys.path.insert(0, LOFTR)
        from src.loftr import LoFTR, default_cfg
        self.model = LoFTR(config=default_cfg)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        sd = {(k.replace("matcher.", "", 1) if k.startswith("matcher.") else k): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.eval().to(device)
        self.device = device

    def _prep(self, gray):
        h, w = gray.shape
        H, W = max(32, (h // 32) * 32), max(32, (w // 32) * 32)
        t = torch.from_numpy(cv2.resize(gray, (W, H)))[None][None].float().to(self.device) / 255.0
        return t, np.array([w / W, h / H], dtype=np.float32)

    @torch.no_grad()
    def __call__(self, i0, i1):
        t0, s0 = self._prep(cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY))
        t1, s1 = self._prep(cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY))
        b = {"image0": t0, "image1": t1}
        self.model(b)
        return b["mkpts0_f"].cpu().numpy() * s0, b["mkpts1_f"].cpu().numpy() * s1


def build_matcher(a, device="cuda"):
    ckpt = getattr(a, "extra_ckpt", None) or CKPT
    m = LoFTRMatcher(ckpt, device)

    def f(i0, i1):          # closure: complexity/bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or "loftr_outdoor")
