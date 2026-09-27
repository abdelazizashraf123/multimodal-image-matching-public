"""RoMa v2 ("RoMa v2: Harder Better Faster Denser Feature Matching", arXiv 2511.15706,
github.com/Parskatt/RoMaV2), the successor of the RoMa we evaluate as "RoMa v1".

Run exactly as the repository does: `RoMaV2()` in its default setting "precise" (both images
resized to 800 x 800 for the coarse stage and 1280 x 1280 for the refinement, bidirectional), then
`sample(preds, N)` with the authors' balanced sampling and `to_pixel_coordinates`. N = 10,000 to
match the v1 protocol (v1 samples 10,000 as well); the README example uses 5,000. Other settings
can be selected with --romav2_setting: "mega1500" is the paper's benchmark setting (800 coarse,
1024 refinement, bidirectional, overlap threshold 0.05), the closest preset to v1's structure
(professor's choice, 2026-09-17); "turbo" 320 / "fast" 512 / "base" 640 have no refinement stage.
The encoder is DINOv3 ViT-L/16 whose code the repo pulls through torch.hub at a pinned commit and
whose weights sit inside the release checkpoint romav2.0.1.pt; both live in $TORCH_HOME/hub and
must be pre-fetched on the login node. The model refuses to run unless float32 matmul precision is
"highest", so it is set here.
FLOPs: match() and forward() carry @torch.inference_mode(), which hides ops from the FLOP counter;
while a FlopCounterMode is active the undecorated functions are called under no_grad (same ops).
"""
import sys
import types

import numpy as np
import torch

ROMAV2 = "/cta/users/h-abdelaziz/projects/RoMaV2"
SETTINGS = ("precise", "mega1500", "base", "fast", "turbo")


def _flop_counter_active():
    from torch.utils.flop_counter import FlopCounterMode
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    return any(isinstance(m, FlopCounterMode) for m in _get_current_dispatch_mode_stack())


class RoMaV2Matcher:
    def __init__(self, setting="precise", num_matches=10000, device="cuda"):
        if setting not in SETTINGS:
            raise ValueError(f"unknown RoMa v2 setting {setting!r}; known: {SETTINGS}")
        torch.set_float32_matmul_precision("highest")
        if ROMAV2 + "/src" not in sys.path:
            sys.path.insert(0, ROMAV2 + "/src")
        from romav2 import RoMaV2
        self.model = RoMaV2(RoMaV2.Cfg(setting=setting)).eval()   # hub weights + DINOv3 code from $TORCH_HOME/hub
        self.setting, self.n, self.device = setting, num_matches, device

    @torch.no_grad()
    def __call__(self, i0, i1):
        """BGR uint8 arrays -> (kpts0, kpts1) pixel coordinates in the original frames."""
        A = np.ascontiguousarray(i0[:, :, ::-1])       # RGB HxWx3 uint8, what _load_image expects
        B = np.ascontiguousarray(i1[:, :, ::-1])
        m = self.model
        if _flop_counter_active():
            m.forward = types.MethodType(type(m).forward.__wrapped__, m)   # bypass @inference_mode on forward()
            try:
                preds = type(m).match.__wrapped__(m, A, B)                  # ... and on match()
            finally:
                del m.forward
        else:
            preds = m.match(A, B)
        matches, _conf, _pA, _pB = m.sample(preds, self.n)
        k0, k1 = m.to_pixel_coordinates(matches, A.shape[0], A.shape[1], B.shape[0], B.shape[1])
        return k0.float().cpu().numpy(), k1.float().cpu().numpy()


def build_matcher(a, device="cuda"):
    setting = getattr(a, "romav2_setting", None) or "precise"
    m = RoMaV2Matcher(setting, 10000, device)

    def f(i0, i1):          # closure: complexity/bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or f"romav2_{setting}")
