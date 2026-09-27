# VERBATIM copy of /cta/users/a-shahzeen/mast3r_dense.py (her MASt3R fast-path adapter, 2026-09-15 version), taken
# 2026-09-24; the only edit is the repository path on the sys.path line. Kept verbatim so the fast variant runs here
# exactly as in her measurements (fast_cfg read from the checkpoint: attention, autocast scope, desc_only, prep).
"""MASt3R as a drop-in dense matcher for the LiDAR-correspondence PCK benchmarks.

The LiDAR protocol asks "where does THIS pixel go?" for arbitrary query points, but
MASt3R natively returns sparse reciprocal-NN matches at points of its own choosing, so
fast_reciprocal_NNs cannot answer it. What MASt3R does expose is a per-pixel descriptor
map for each image, and that IS queryable: sample the query pixel's descriptor from
image 0's map, correlate it against every position in image 1's map, take the peak.

RESOLUTION HANDICAP -- state this whenever these numbers sit beside RoMa's. MASt3R runs
at a 512 px long edge, so on a 1280 px source the descriptor grid is 2.5x coarser than
the image. A raw argmax therefore quantises to ~2.5 px, which alone would cost most of
PCK@1 and a chunk of PCK@3. We recover sub-grid accuracy with a local softmax centroid
around the peak, but MASt3R is still structurally disadvantaged at tight thresholds
versus RoMa's 864 px upsampled warp. Read @10/@20 for the fairer comparison.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as tvf

sys.path.insert(0, "/cta/users/h-abdelaziz/projects/mast3r_fast")   # copy of a-shahzeen/mast3r (her fast path): jobs/setup_mast3r_fast.sh
import mast3r.utils.path_to_dust3r  # noqa: E402,F401
from mast3r.model import AsymmetricMASt3R  # noqa: E402
from dust3r.inference import inference  # noqa: E402  (kept for the reference path)

AMP_DTYPES = {None: None, "none": None, "fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16,
              "fp16enc": torch.float16}   # 'fp16enc': autocast the encoder only, decoder+heads stay TF32/fp32

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class MASt3RDense:
    """Exposes .warp_at(img0_bgr, img1_bgr, pts0) -> pts1, matching RoMaDense's signature."""

    def __init__(self, ckpt_path, device="cuda", size=512, win=2, amp=None, desc_only=None, prep=None):
        """amp: None/'fp32' | 'fp16' | 'bf16' autocast for encoder+decoder (heads always fp32, as
        in dust3r's forward). desc_only: skip the DPT pts3d branch, which the matcher never reads.
        Both default to the checkpoint's own 'fast' metadata when present, else off. Env
        MAST3R_AMP / MAST3R_DESC_ONLY override the checkpoint; explicit args override everything."""
        self.device = device
        self.size = size
        self.win = win          # half-width of the sub-grid refinement window
        self.model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device).eval()
        fast = getattr(self.model, "fast_cfg", None) or {}
        import os
        if amp is None:
            amp = os.environ.get("MAST3R_AMP", fast.get("autocast"))
        if desc_only is None:
            e = os.environ.get("MAST3R_DESC_ONLY")
            desc_only = (e == "1") if e is not None else bool(fast.get("desc_only", False))
        self.amp_dtype = AMP_DTYPES[amp]
        self.amp_scope = "encoder" if (amp or "").endswith("enc") else "all"
        self.desc_only = bool(desc_only)
        # 'pil' = original PIL LANCZOS resize on the CPU (~38 ms/pair); 'gpu' = antialiased bicubic
        # resize of the uint8 image on the GPU.
        self.prep = prep or os.environ.get("MAST3R_PREP", fast.get("prep", "pil"))
        # dust3r.training sets this at import time, so every published MASt3R number was TF32;
        # make it explicit rather than import-order dependent.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        for head in (self.model.downstream_head1, self.model.downstream_head2):
            head.desc_only = self.desc_only
        print(f"MASt3RDense: attn={__import__('models.blocks', fromlist=['ATTN_IMPL']).ATTN_IMPL} "
              f"amp={amp} desc_only={self.desc_only} prep={self.prep}", flush=True)

    def _prep(self, img_bgr, idx):
        h, w = img_bgr.shape[:2]
        long_edge = max(w, h)
        nw = max(16, (int(round(w * self.size / long_edge)) // 16) * 16)
        nh = max(16, (int(round(h * self.size / long_edge)) // 16) * 16)
        if self.prep == "gpu":
            t = torch.from_numpy(img_bgr).to(self.device, non_blocking=True)      # H,W,3 uint8 BGR
            t = t.flip(-1).permute(2, 0, 1)[None].float()                       # 1,3,H,W RGB
            t = F.interpolate(t, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
            t = (t.clamp_(0, 255) / 127.5 - 1.0)                                # == Normalize(.5,.5)
        else:
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            t = ImgNorm(pil.resize((nw, nh), Image.LANCZOS))[None]
        view = dict(img=t, true_shape=np.int32([[nh, nw]]), idx=idx, instance=str(idx))
        return view, (w / nw, h / nh), (nw, nh)

    @torch.no_grad()
    def _forward(self, view1, view2):
        m = self.model
        if self.amp_dtype is None or self.amp_scope == "all":
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                return m(view1, view2)
        # encoder-only autocast: same sequence as AsymmetricCroCo3DStereo.forward, but the
        # decoder (cross-attention, cast-overhead-bound at these token counts) stays in fp32/TF32
        with torch.autocast("cuda", dtype=self.amp_dtype):
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = m._encode_symmetrized(view1, view2)
        dec1, dec2 = m._decoder(feat1.float(), pos1, feat2.float(), pos2)
        res1 = m._downstream_head(1, [t.float() for t in dec1], shape1)
        res2 = m._downstream_head(2, [t.float() for t in dec2], shape2)
        if 'pts3d' in res2:
            res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return res1, res2

    @torch.no_grad()
    def warp_at(self, img0_bgr, img1_bgr, pts0):
        v0, (sx0, sy0), (nw0, nh0) = self._prep(img0_bgr, 0)
        v1, (sx1, sy1), (nw1, nh1) = self._prep(img1_bgr, 1)
        # direct forward (dust3r's inference() disables autocast internally and round-trips
        # every output through the CPU; neither is wanted on the latency path)
        def to_dev(v):
            return dict(img=v["img"].to(self.device, non_blocking=True),
                        true_shape=torch.from_numpy(v["true_shape"]), idx=[v["idx"]], instance=[v["instance"]])
        pred1, pred2 = self._forward(to_dev(v0), to_dev(v1))
        d0 = pred1["desc"].squeeze(0).detach().float()   # (H0,W0,D) -- heads run in fp32
        d1 = pred2["desc"].squeeze(0).detach().float()   # (H1,W1,D)
        H0, W0, D = d0.shape
        H1, W1, _ = d1.shape

        # query descriptors: bilinear sample at the query pixels, in descriptor-grid coords
        q = torch.from_numpy(np.asarray(pts0, dtype=np.float32)).to(self.device)
        gx = (q[:, 0] / sx0) / max(nw0 - 1, 1) * (W0 - 1)
        gy = (q[:, 1] / sy0) / max(nh0 - 1, 1) * (H0 - 1)
        grid = torch.stack([gx / max(W0 - 1, 1) * 2 - 1, gy / max(H0 - 1, 1) * 2 - 1], -1)
        qd = F.grid_sample(d0.permute(2, 0, 1)[None], grid[None, None],
                           mode="bilinear", align_corners=True)[0, :, 0].T   # (N,D)

        flat = d1.reshape(-1, D)                                   # (H1*W1, D)
        N = len(qd)
        best = torch.empty(N, dtype=torch.long, device=self.device)
        # chunk the correlation: N x H1*W1 would be ~1.6 GB at N=2000, H1*W1=200k
        step = max(1, int(4e7 // max(flat.shape[0], 1)))
        for s in range(0, N, step):
            best[s:s + step] = (qd[s:s + step] @ flat.T).argmax(1)
        by = (best // W1).float()
        bx = (best % W1).float()

        # sub-grid refinement: softmax centroid over a (2*win+1)^2 window about the peak,
        # which recovers most of what the 2.5x-coarse descriptor grid throws away
        if self.win > 0:
            offs = torch.arange(-self.win, self.win + 1, device=self.device)
            oy, ox = torch.meshgrid(offs, offs, indexing="ij")
            oy = oy.reshape(-1).float(); ox = ox.reshape(-1).float()
            wy = (by[:, None] + oy[None]).clamp(0, H1 - 1)
            wx = (bx[:, None] + ox[None]).clamp(0, W1 - 1)
            idx = (wy.long() * W1 + wx.long())                     # (N,K)
            nbr = flat[idx.reshape(-1)].reshape(N, idx.shape[1], D)
            sim = torch.einsum("nd,nkd->nk", qd, nbr)
            wgt = torch.softmax(sim * 10.0, dim=1)
            by = (wgt * wy).sum(1)
            bx = (wgt * wx).sum(1)

        # descriptor grid -> model-input px -> original image px
        px = bx / max(W1 - 1, 1) * max(nw1 - 1, 1) * sx1
        py = by / max(H1 - 1, 1) * max(nh1 - 1, 1) * sy1
        return torch.stack([px, py], -1).cpu().numpy()
