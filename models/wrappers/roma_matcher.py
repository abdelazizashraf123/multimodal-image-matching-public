"""Original RoMa (released roma_outdoor.pth) with the group's reference settings.

Settings are copied from a-shahzeen's wrappers (dsert/eval_dsert_pose.py and
complexity_scripts/eval_rsmd_test.py) so the numbers are comparable with the published tables:
  * coarse 560x560, upsample 864x864, fp16 autocast, attenuate_cert=True
  * use_custom_corr=False -- the fused CUDA kernel is an optional extra and is NOT installed;
    without this flag this RoMa version crashes with "No module named 'local_corr'"
  * 10,000 sampled matches (romatch default, sample_mode="threshold_balanced")

`romatch` resolves to the editable install of /cta/users/h-abdelaziz/projects/RoMa (the clean
upstream clone, commit 77f8d68). Weights are read from $TORCH_HOME/hub/checkpoints in scratch,
never from the network.
"""
import os
import cv2
import torch
from PIL import Image
from romatch.models.model_zoo.roma_models import roma_model

HUB = os.path.join(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
                   "hub", "checkpoints")
PRETRAINED = os.path.join(HUB, "roma_outdoor.pth")
DINOV2 = os.path.join(HUB, "dinov2_vitl14_pretrain.pth")


class RoMaMatcher:
    def __init__(self, pretrained=PRETRAINED, dinov2=DINOV2, device="cuda",
                 coarse_res=560, upsample_res=864, finetuned=None):
        self.model = roma_model(resolution=(coarse_res, coarse_res), upsample_preds=True,
                                device=device,
                                weights=torch.load(pretrained, map_location="cpu"),
                                dinov2_weights=torch.load(dinov2, map_location="cpu"),
                                amp_dtype=torch.float16, use_custom_corr=False,
                                attenuate_cert=True,
                                upsample_res=(upsample_res, upsample_res))
        if finetuned:
            self.model.load_state_dict(torch.load(finetuned, map_location="cpu")["model"])
        self.model.eval()
        self.device = device

    @staticmethod
    def _flop_counter_active():
        from torch.utils.flop_counter import FlopCounterMode
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
        return any(isinstance(m, FlopCounterMode) for m in _get_current_dispatch_mode_stack())

    @torch.no_grad()
    def __call__(self, i0, i1):
        """BGR uint8 arrays (as returned by cv2.imread) -> (kpts0, kpts1) pixel coordinates."""
        A = Image.fromarray(cv2.cvtColor(i0, cv2.COLOR_BGR2RGB))
        B = Image.fromarray(cv2.cvtColor(i1, cv2.COLOR_BGR2RGB))
        if self._flop_counter_active():
            # This RoMa version decorates match() with @torch.inference_mode(), and inference
            # mode hides every op from torch.utils.flop_counter (counts come back as 0). Only
            # while a FlopCounterMode is active, call the undecorated function under no_grad:
            # same ops, same numbers, just observable. Timing and evaluation keep the real path.
            match_fn = type(self.model).match.__wrapped__
            warp, cert = match_fn(self.model, A, B, batched=True, device=self.device)
        else:
            warp, cert = self.model.match(A, B, batched=True, device=self.device)
        s, _ = self.model.sample(warp, cert)
        k0, k1 = self.model.to_pixel_coordinates(s, A.size[1], A.size[0], B.size[1], B.size[0])
        return k0.cpu().numpy(), k1.cpu().numpy()


def build_matcher(a, device):
    """Same contract as a-shahzeen's eval_rsmd_test.build_matcher: returns (callable, tag).

    Used by complexity/bench_*.py. The callable is a closure rather than a bound method on
    purpose: bench_complexity.modules_in() recovers the nn.Modules from fn.__closure__.
    """
    matcher = getattr(a, "matcher", None) or "roma"
    if matcher != "roma":
        raise ValueError(f"this eval folder only builds original RoMa, got --matcher {matcher}")
    finetuned = getattr(a, "finetuned", None)
    m = RoMaMatcher(getattr(a, "pretrained", None) or PRETRAINED,
                    getattr(a, "dinov2", None) or DINOV2, device, finetuned=finetuned)

    def f(i0, i1):
        return m(i0, i1)

    tag = getattr(a, "tag", None) or (
        "roma_outdoor" if not finetuned else "roma_" + os.path.basename(os.path.dirname(finetuned)))
    return f, tag
