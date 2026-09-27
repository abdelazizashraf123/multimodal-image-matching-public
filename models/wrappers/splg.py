"""MINIMA-LightGlue: SuperPoint + LightGlue with MINIMA's retrained LightGlue weights, the `sp_lg` arm of the
MINIMA repository and of Shahzeen Ahmad's RSMD table. Used here for the encoder / decoder / head profile
(complexity/run_modprof.py --matcher splg), so that the model's row in the report has the same measured columns
as every other row.

Fidelity: the model and the image path are MINIMA's, transcribed line for line from load_model.load_sp_lg and
src/utils/data_io_sp_lg.DataIOWrapper (MINIMA's default test config), without importing MINIMA's config
machinery (yacs), so the wrapper runs in an environment that has the FLOP counter (myenv, torch 2.5; MINIMA's
own environment is torch 2.0.1 and counts no FLOPs):
  * SuperPoint: 2,048 keypoints, NMS radius 4, detection threshold 5e-4, border 4 (its own weights);
  * LightGlue: 9 layers, 4 heads, flash attention, depth / width confidence 0.95 / 0.99, filter threshold 0.1,
    built as LightGlue(features="superpoint") and then overwritten with MINIMA's checkpoint after the same key
    renaming MINIMA applies;
  * images: grayscale, long edge resized to 640 (rounded), sides floored to a multiple of 8, /255; SuperPoint's
    extract() then rescales the long edge to 1024 itself, which is why her measured working resolution is
    1024x576 for a 1620x925 source (640x360 -> 1024x576); keypoints are scaled back to the original frame.

Weights (login node, compute nodes have no internet):
    /cta/scratch/h-abdelaziz/weights/minima/minima_lightglue.pth      (MINIMA release, LSXI7/storage)
    $TORCH_HOME/hub/checkpoints/superpoint_v1.pth                        (cvg/LightGlue v0.1_arxiv)
    $TORCH_HOME/hub/checkpoints/superpoint_lightglue.pth                 (cvg/LightGlue v0.1_arxiv)
LightGlue asks torch.hub for the last two under other file names; the shim below serves them from the cache
directory by their release names so nothing is downloaded on a compute node.
"""
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch

MINIMA = "/cta/users/h-abdelaziz/projects/MINIMA"
CKPT = "/cta/scratch/h-abdelaziz/weights/minima/minima_lightglue.pth"
SP_CONF = dict(descriptor_dim=256, nms_radius=4, max_num_keypoints=2048, detection_threshold=0.0005, remove_borders=4)
LG_CONF = dict(name="lightglue", input_dim=256, descriptor_dim=256, add_scale_ori=False, n_layers=9, num_heads=4, flash=True,
               mp=False, depth_confidence=0.95, width_confidence=0.99, filter_threshold=0.1, weights=None)
RESIZE, DF = 640, 8          # MINIMA src/config/default.py: TEST.IMG0_RESIZE / TEST.DF, no padding


def _serve_hub_weights_locally():
    """torch.hub.load_state_dict_from_url -> the file of the same basename in $TORCH_HOME/hub/checkpoints, if present."""
    import torch.hub as H
    if getattr(H, "_splg_local_shim", False):
        return
    orig = H.load_state_dict_from_url

    def local(url, *a, **k):
        cache = osp.join(os.environ.get("TORCH_HOME", osp.expanduser("~/.cache/torch")), "hub", "checkpoints")
        fn = osp.join(cache, osp.basename(url))
        if osp.exists(fn):
            return torch.load(fn, map_location=k.get("map_location", "cpu"))
        return orig(url, *a, **k)

    H.load_state_dict_from_url = local
    H._splg_local_shim = True


class Matching(torch.nn.Module):
    """MINIMA's Matching module (load_model.load_sp_lg), same child names: .extractor, .matcher."""

    def __init__(self, ckpt, device):
        super().__init__()
        if MINIMA not in sys.path:
            sys.path.insert(0, MINIMA)
        _serve_hub_weights_locally()
        from third_party.LightGlue.lightglue import LightGlue, SuperPoint
        from third_party.LightGlue.lightglue.utils import rbd
        self._rbd = rbd
        self.extractor = SuperPoint(**SP_CONF).eval().to(device)
        self.matcher = LightGlue(features="superpoint", **LG_CONF).eval().to(device)
        sd = torch.load(ckpt, map_location=device)
        for i in range(LG_CONF["n_layers"]):                     # MINIMA's renaming of old checkpoint keys
            for old, new in ((f"self_attn.{i}", f"transformers.{i}.self_attn"), (f"cross_attn.{i}", f"transformers.{i}.cross_attn")):
                sd = {k.replace(old, new): v for k, v in sd.items()}
        missing, unexpected = self.matcher.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"MINIMA LightGlue checkpoint has unexpected keys: {list(unexpected)[:5]}")

    def forward(self, batch):
        feats0 = self.extractor.extract(batch["image0"])        # auto-resizes the long edge to 1024, as MINIMA runs it
        feats1 = self.extractor.extract(batch["image1"])
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [self._rbd(x) for x in (feats0, feats1, matches01)]
        m = matches01["matches"]
        return feats0["keypoints"][m[..., 0]], feats1["keypoints"][m[..., 1]]


class SPLGMatcher:
    def __init__(self, ckpt=CKPT, device="cuda"):
        if not osp.exists(ckpt):
            raise FileNotFoundError(f"MINIMA LightGlue weights not found: {ckpt} (see the module docstring)")
        self.model = Matching(ckpt, device)
        self.device = device

    def _prep(self, bgr):
        """DataIOWrapper.preprocess_image with MINIMA's test defaults; returns the tensor and the back-scale."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        h, w = gray.shape[:2]
        s = RESIZE / max(h, w)
        w2, h2 = int(round(w * s)), int(round(h * s))
        w2, h2 = int(w2 // DF * DF), int(h2 // DF * DF)
        img = cv2.resize(gray, (w2, h2))
        t = torch.from_numpy(img)[None][None].to(self.device).float() / 255.0
        return t, np.array([w / w2, h / h2], dtype=np.float32)

    @torch.no_grad()
    def __call__(self, i0, i1):
        t0, s0 = self._prep(i0); t1, s1 = self._prep(i1)
        k0, k1 = self.model({"image0": t0, "image1": t1})
        return k0.cpu().numpy() * s0, k1.cpu().numpy() * s1


def build_matcher(a, device="cuda"):
    m = SPLGMatcher(getattr(a, "extra_ckpt", None) or CKPT, device)

    def f(i0, i1):          # closure: complexity/bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or "splg")
