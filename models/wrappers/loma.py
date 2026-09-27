"""LoMa-B ("LoMa: Local Feature Matching Revisited", ECCV 2026, github.com/davnords/LoMa), used
for the per-dataset complexity row only (professor, 2026-09-16). LightGlue-style sparse matcher:
frozen DaD detector + frozen DeDoDe-G descriptor (DINOv2 ViT-L inside) + a 9-layer transformer.

The authors' `LoMa.match(path_A, path_B)` loads each image twice with different preprocessing.
This wrapper replicates that path route from in-memory BGR arrays (same as a-shahzeen's
loma_matcher.py, which produced the group's LoMa rows):
  * detector  (DaD):      RGB, PIL resize to 1024 on the long edge, dims floored to a multiple of 8
  * descriptor (DeDoDe-G): RGB, PIL resize to a fixed 784 x 784
  * keypoints are normalised to [-1, 1]; they are mapped back with the ORIGINAL array size
  * matcher forward under bf16 autocast (cfg.mp), 2,048 keypoints, filter threshold 0.1
FLOPs: LoMa.detect/describe/match and FrozenDINOv2.forward run under torch.inference_mode, which
hides ops from torch.utils.flop_counter. We call the module forwards directly under no_grad and,
only while a FlopCounterMode is active, swap torch.inference_mode for torch.no_grad so the DINOv2
pass inside the descriptor is counted as well (a-shahzeen's 7,700 GFLOPs row includes it). Same
ops, same numbers; timing and evaluation take the plain path.
Weights: the repo downloads them itself into $TORCH_HOME/hub/checkpoints (pre-fetched on the login
node because compute nodes have no internet): loma_B.pt, dad.pth, dedode_descriptor_G.pth,
dinov2_vitl14_pretrain.pth.
"""
import contextlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LOMA = "/cta/users/h-abdelaziz/projects/LoMa"
VARIANTS = ("loma_B", "loma_B128", "loma_L", "loma_G", "loma_R")


def _flop_counter_active():
    from torch.utils.flop_counter import FlopCounterMode
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    return any(isinstance(m, FlopCounterMode) for m in _get_current_dispatch_mode_stack())


@contextlib.contextmanager
def _inference_mode_as_no_grad():
    """FrozenDINOv2.forward opens `torch.inference_mode()` at call time; make it a no_grad so the
    FLOP counter sees the ViT. Restored on exit."""
    orig = torch.inference_mode
    torch.inference_mode = torch.no_grad
    try:
        yield
    finally:
        torch.inference_mode = orig


class LoMaMatcher:
    def __init__(self, variant="loma_B", device="cuda"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown LoMa variant {variant!r}; known: {VARIANTS}")
        if LOMA + "/src" not in sys.path:
            sys.path.insert(0, LOMA + "/src")
        import loma
        from loma.loma import filter_matches, to_pixel_coords
        cfg_cls = {"loma_B": loma.LoMaB, "loma_B128": loma.LoMaB128, "loma_L": loma.LoMaL,
                   "loma_G": loma.LoMaG, "loma_R": loma.LoMaR}[variant]
        self.cfg = cfg_cls(compile=False)          # torch.compile off: irrelevant to the score, slow to build
        self.model = loma.LoMa(self.cfg).eval()    # loads cfg.weights_url through torch.hub, moves to cuda
        self.device = device
        self.variant = variant
        self._filter, self._to_pixel = filter_matches, to_pixel_coords
        det = self.model._detector
        self.det_resize, self.det_keep_ar = det.resize, det.keep_aspect_ratio     # 1024, True
        self.desc_hw = (784, 784)                                               # describe_keypoints_from_path default

    def _det_tensor(self, pil):                    # DaD.load_image
        W, H = pil.size
        if self.det_keep_ar:
            s = self.det_resize / max(W, H)
            W, H = int((s * W) // 8 * 8), int((s * H) // 8 * 8)
        else:
            H = W = self.det_resize
        a = np.array(pil.resize((W, H))) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).float().to(self.device)[None]

    def _desc_tensor(self, pil):                   # DeDoDeDescriptor.read_image
        H, W = self.desc_hw
        a = np.array(pil.resize((W, H))) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).float().to(self.device)[None]

    def _detect_describe(self, bgr):
        pil = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))          # RGB, as Image.open(...).convert("RGB")
        kpts = self.model._detector(self._det_tensor(pil), self.cfg.num_keypoints)["keypoints"]   # == DaD.detect() minus the decorator
        grid = self.model._descriptor(self._desc_tensor(pil))                                     # == describe_keypoints() body
        desc = F.grid_sample(grid.float(), kpts[:, None], mode="bilinear", align_corners=False)[:, :, 0].mT
        h, w = bgr.shape[:2]
        return kpts, desc, h, w

    @torch.no_grad()
    def __call__(self, i0, i1):
        """BGR uint8 arrays -> (kpts0, kpts1) float32 pixel coordinates in the original frames."""
        ctx = _inference_mode_as_no_grad() if _flop_counter_active() else contextlib.nullcontext()
        with ctx:
            k0, d0, h0, w0 = self._detect_describe(i0)
            k1, d1, h1, w1 = self._detect_describe(i1)
            scores = self.model(k0, k1, d0, d1)["scores"]       # LoMa.forward: bf16 autocast inside
        m0, _, _, _ = self._filter(scores, self.cfg.filter_threshold)
        valid = m0[0] > -1
        if not bool(valid.any()):
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        a = self._to_pixel(k0[0][torch.where(valid)[0]], h0, w0).float().cpu().numpy()
        b = self._to_pixel(k1[0][m0[0][valid]], h1, w1).float().cpu().numpy()
        return a.astype(np.float32), b.astype(np.float32)


def build_matcher(a, device="cuda"):
    variant = getattr(a, "loma_variant", None) or "loma_B"
    m = LoMaMatcher(variant, device)

    def f(i0, i1):          # closure: complexity/bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or variant)
