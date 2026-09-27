"""MASt3R-fast: Shahzeen Ahmad's low-cost inference path for MASt3R (same weights, no retraining), run under the
group's dense-grid protocol exactly like matchers/mast3r.py mode="grid", so its rows are comparable with the
MASt3R rows in every table.

What "fast" is (her fastckpt_smoke/RESULTS.md): fused scaled-dot-product attention in the CroCo blocks, a
vectorised rotary-embedding fallback, fp16 autocast over the encoder only, the DPT 3D-point head skipped
(desc_only), and the image resize done on the GPU. All of it is switched on by the `fast` metadata block stored
in her fast checkpoint and read by her adapter (matchers/mast3r_dense_fast.py, a verbatim copy). The code is a
copy of her modified repository (jobs/setup_mast3r_fast.sh -> ~/projects/mast3r_fast); the checkpoint is her
MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_fast.pth (fp16-stored weights, 688.6 M parameters, identical
tensors to the release file).
"""
import numpy as np

CKPT = "/cta/scratch/h-abdelaziz/weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_fast.pth"


class MASt3RFastGrid:
    """a-shahzeen's pose protocol: 60x40 query grid (2,400 points) through her fast adapter's warp_at."""

    def __init__(self, ckpt=CKPT, device="cuda"):
        from matchers.mast3r_dense_fast import MASt3RDense
        self.dense = MASt3RDense(ckpt, device)          # amp / desc_only / prep from the checkpoint's fast metadata
        self.model = self.dense.model                    # for bench_complexity.modules_in()
        fast = getattr(self.model, "fast_cfg", None)
        if not fast:
            raise RuntimeError(f"{ckpt} carries no 'fast' metadata: this is not the fast checkpoint")

    def __call__(self, i0, i1):
        h, w = i0.shape[:2]
        xs, ys = np.meshgrid(np.linspace(20, w - 20, 60), np.linspace(20, h - 20, 40))
        p0 = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32)
        return p0, self.dense.warp_at(i0, i1, p0)


def build_matcher(a, device="cuda"):
    m = MASt3RFastGrid(getattr(a, "extra_ckpt", None) or CKPT, device)

    def f(i0, i1):          # closure: complexity/bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or "mast3r_fast")
