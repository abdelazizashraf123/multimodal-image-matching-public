"""MASt3R fast inference path -- same weights and same protocol as `mast3r`, lower latency.

NOT a reduced or distilled model: the checkpoint holds the identical tensors and parameter
count as MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth, optionally stored in fp16. It
runs the same 60x40 = 2,400-point dense-query grid as `mast3r` (mode="grid"), so the two arms
are directly comparable; only the inference configuration differs.

Three inference-side changes, in descending order of what they save:

  1. GPU-side resize. The image is resized on the device with antialiased bicubic instead of a
     host-side PIL LANCZOS resize. Same geometry (long edge 512, dims floored to a multiple of
     16) and the same (x/127.5 - 1) normalisation. This is the largest single contribution, and
     being a data-path change it applies equally to the unmodified model.
  2. Skip the DPT pts3d branch. The matcher reads `desc` / `desc_conf` and never consumes
     `pts3d`, which is otherwise computed and discarded. The only change of the three that
     reduces FLOPs.
  3. fp16 encoder autocast. The ViT-L encoder runs in fp16; the decoder and prediction heads
     stay fp32/TF32, because at these token counts the decoder is cast-overhead bound.

Peak GPU memory RISES slightly, because autocast holds activations in two precisions: this
trades memory for speed rather than reducing the footprint. Accuracy is very close to the base
arm; the small difference is fp16 precision, not the skipped branch.

Note when comparing against older timings: fused SDPA attention later became the default for
every MASt3R arm, so a figure measured before that change is not a like-for-like baseline for
this one. Both arms here run SDPA.

SELF-CONTAINED ON A STOCK CLONE. Two of the three changes need behaviour the official MASt3R /
croco source does not expose (an SDPA attention path, and a way to tell the head to skip DPT).
Some forks carry those as source patches. Rather than depend on a patched clone, this wrapper
installs both at run time and reports which are actually active -- so it can never quietly fall
back to a different operating point while still calling itself "fast".
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as tvf

# Prefer a clone sitting inside the repo (models/mast3r); fall back to the absolute path the
# other wrappers use, so this file works unchanged on either layout.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAST3R = (os.path.join(_REPO, "models", "mast3r")
          if os.path.isdir(os.path.join(_REPO, "models", "mast3r", "mast3r"))
          else "/cta/users/h-abdelaziz/projects/mast3r")
CKPT = ("/cta/scratch/h-abdelaziz/weights/mast3r/"
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _import_mast3r():
    if MAST3R not in sys.path:
        sys.path.insert(0, MAST3R)
    import mast3r.utils.path_to_dust3r  # noqa: F401  (puts dust3r on the path)
    from mast3r.model import AsymmetricMASt3R
    return AsymmetricMASt3R


def _install_sdpa():
    """Swap croco's hand-rolled attention for fused scaled_dot_product_attention.

    Mathematically identical; SDPA picks a fused kernel and avoids materialising the N x N
    attention matrix. Returns True if the patch went in, False if the source already provides
    it (a fork exposing CROCO_ATTN) or the class could not be found.
    """
    try:
        from models import blocks as croco_blocks
    except Exception:
        return False
    if getattr(croco_blocks, "ATTN_IMPL", None) == "sdpa":
        return True                      # patched clone already defaults to SDPA
    Attention = getattr(croco_blocks, "Attention", None)
    if Attention is None or getattr(Attention, "_sdpa_patched", False):
        return bool(Attention is not None)

    def forward(self, x, xpos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        x = F.scaled_dot_product_attention(q, k, v)          # fused; no explicit softmax
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    Attention.forward = forward
    Attention._sdpa_patched = True
    return True


def _install_desc_only(model):
    """Make the two downstream heads skip the DPT pts3d branch.

    The matcher reads `desc` / `desc_conf`; `pts3d` is computed and discarded. The official
    Cat_MLP_LocalFeatures_DPT_Pts3d head has no switch for this, so replace its forward with
    one that reproduces the official code path minus the `self.dpt(...)` call. Returns True if
    both heads are now skipping, False if this build does not look like that head (in which
    case the caller warns rather than silently reporting a "fast" run that is not).
    """
    heads = [model.downstream_head1, model.downstream_head2]
    if not all(hasattr(h, "dpt") and hasattr(h, "head_local_features") for h in heads):
        return False

    cls = type(heads[0])
    # a patched clone may already honour desc_only in its own source; wrapping it
    # would work but would run our reimplementation instead of theirs, so leave it alone.
    import inspect
    try:
        native = "desc_only" in inspect.getsource(cls.forward)
    except (OSError, TypeError):
        native = False
    if not native and not getattr(cls, "_desc_only_patched", False):
        inner = cls.forward
        from mast3r.catmlp_dpt_head import reg_desc
        from dust3r.heads.postprocess import reg_dense_conf

        def forward(self, decout, img_shape, _inner=inner):
            if not getattr(self, "desc_only", False):
                return _inner(self, decout, img_shape)
            # identical to the official forward with `pts3d = self.dpt(...)` removed
            cat_output = torch.cat([decout[0], decout[-1]], dim=-1)
            H, W = img_shape
            B = cat_output.shape[0]
            lf = self.head_local_features(cat_output)
            lf = lf.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
            lf = F.pixel_shuffle(lf, self.patch_size)                     # B,d,H,W
            fmap = lf.permute(0, 2, 3, 1)                                 # B,H,W,d
            res = dict(desc=reg_desc(fmap[..., :self.local_feat_dim], mode=self.desc_mode))
            if self.two_confs:
                res["desc_conf"] = reg_dense_conf(fmap[..., self.local_feat_dim],
                                                  mode=self.desc_conf_mode)
            return res

        cls.forward = forward
        cls._desc_only_patched = True

    for h in heads:
        h.desc_only = True
    return True


class MASt3RFastDense:
    """`mast3r`'s dense-query protocol on the fast inference path.

    Differences from wrappers/mast3r.py MASt3RDense, all inference-side:
      * prep="gpu"   : resize the uint8 image on the device with antialiased bicubic instead of
                       a host-side PIL LANCZOS resize. Same geometry (long edge 512, dims floored
                       to a multiple of 16) and the same (x/127.5 - 1) normalisation.
      * amp="fp16enc": autocast the ViT-L encoder to fp16; decoder and heads stay fp32/TF32,
                       because at these token counts the decoder is cast-overhead bound.
      * desc_only    : skip the DPT pts3d branch.
    It also calls the model directly rather than through dust3r's `inference()`, which disables
    autocast internally and round-trips every output through the CPU.
    """

    def __init__(self, ckpt_path=CKPT, device="cuda", size=512, win=2,
                 prep="gpu", amp="fp16enc", desc_only=True):
        AsymmetricMASt3R = _import_mast3r()
        self.device, self.size, self.win = device, size, win
        self.prep = os.environ.get("MAST3R_PREP", prep)
        amp = os.environ.get("MAST3R_AMP", amp)
        self.amp_dtype = {"fp16enc": torch.float16, "fp16": torch.float16,
                          "bf16": torch.bfloat16, "fp32": None, "none": None,
                          None: None}[amp]
        self.amp_encoder_only = (amp == "fp16enc")
        self.model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device).eval()

        # dust3r.training flips these at import time, so every published MASt3R number was
        # already TF32; set them explicitly rather than depend on import order.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.sdpa = _install_sdpa()
        self.desc_only = _install_desc_only(self.model) if desc_only else False
        print(f"MASt3RFast: sdpa={self.sdpa} amp={amp} desc_only={self.desc_only} "
              f"prep={self.prep}", flush=True)
        if desc_only and not self.desc_only:
            print("  note: this MASt3R build exposes no seam to skip the DPT branch; running "
                  "with it enabled (slower, and the DPT branch still costs FLOPs)", flush=True)

    def _prep(self, img_bgr, idx):
        h, w = img_bgr.shape[:2]
        long_edge = max(w, h)
        nw = max(16, (int(round(w * self.size / long_edge)) // 16) * 16)
        nh = max(16, (int(round(h * self.size / long_edge)) // 16) * 16)
        if self.prep == "gpu":
            t = torch.from_numpy(np.ascontiguousarray(img_bgr)).to(self.device, non_blocking=True)
            t = t.flip(-1).permute(2, 0, 1)[None].float()                  # BGR uint8 -> RGB float
            t = F.interpolate(t, size=(nh, nw), mode="bicubic",
                              antialias=True, align_corners=False)
            t = t.clamp_(0, 255) / 127.5 - 1.0                             # == Normalize(.5, .5)
        else:
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            t = ImgNorm(pil.resize((nw, nh), Image.LANCZOS))[None].to(self.device)
        view = dict(img=t, true_shape=torch.from_numpy(np.int32([[nh, nw]])),
                    idx=[idx], instance=[str(idx)])
        return view, (w / nw, h / nh), (nw, nh)

    @torch.no_grad()
    def _forward(self, v0, v1):
        m = self.model
        if self.amp_dtype is None:
            return m(v0, v1)
        if not self.amp_encoder_only:
            with torch.autocast("cuda", dtype=self.amp_dtype):
                return m(v0, v1)
        # encoder-only: same call sequence as AsymmetricCroCo3DStereo.forward, with the decoder
        # and heads left in fp32/TF32
        with torch.autocast("cuda", dtype=self.amp_dtype):
            (s0, s1), (f0, f1), (p0, p1) = m._encode_symmetrized(v0, v1)
        d0, d1 = m._decoder(f0.float(), p0, f1.float(), p1)
        r0 = m._downstream_head(1, [t.float() for t in d0], s0)
        r1 = m._downstream_head(2, [t.float() for t in d1], s1)
        if "pts3d" in r1:
            r1["pts3d_in_other_view"] = r1.pop("pts3d")
        return r0, r1

    @torch.no_grad()
    def warp_at(self, img0_bgr, img1_bgr, pts0):
        """Query pixels in image 0 -> their locations in image 1 (same maths as `mast3r`)."""
        v0, (sx0, sy0), (nw0, nh0) = self._prep(img0_bgr, 0)
        v1, (sx1, sy1), (nw1, nh1) = self._prep(img1_bgr, 1)
        pred0, pred1 = self._forward(v0, v1)
        d0 = pred0["desc"].squeeze(0).detach().float()      # (H0,W0,D)
        d1 = pred1["desc"].squeeze(0).detach().float()      # (H1,W1,D)
        H0, W0, D = d0.shape
        H1, W1, _ = d1.shape

        q = torch.from_numpy(np.asarray(pts0, dtype=np.float32)).to(self.device)
        gx = (q[:, 0] / sx0) / max(nw0 - 1, 1) * (W0 - 1)
        gy = (q[:, 1] / sy0) / max(nh0 - 1, 1) * (H0 - 1)
        grid = torch.stack([gx / max(W0 - 1, 1) * 2 - 1, gy / max(H0 - 1, 1) * 2 - 1], -1)
        qd = F.grid_sample(d0.permute(2, 0, 1)[None], grid[None, None],
                           mode="bilinear", align_corners=True)[0, :, 0].T     # (N,D)

        flat = d1.reshape(-1, D)
        N = len(qd)
        best = torch.empty(N, dtype=torch.long, device=self.device)
        step = max(1, int(4e7 // max(flat.shape[0], 1)))        # chunk: N x H1*W1 is large
        for s in range(0, N, step):
            best[s:s + step] = (qd[s:s + step] @ flat.T).argmax(1)
        by = (best // W1).float()
        bx = (best % W1).float()

        if self.win > 0:      # softmax centroid over a (2*win+1)^2 window about the peak
            offs = torch.arange(-self.win, self.win + 1, device=self.device)
            oy, ox = torch.meshgrid(offs, offs, indexing="ij")
            oy = oy.reshape(-1).float(); ox = ox.reshape(-1).float()
            wy = (by[:, None] + oy[None]).clamp(0, H1 - 1)
            wx = (bx[:, None] + ox[None]).clamp(0, W1 - 1)
            idx = (wy.long() * W1 + wx.long())
            nbr = flat[idx.reshape(-1)].reshape(N, idx.shape[1], D)
            sim = torch.einsum("nd,nkd->nk", qd, nbr)
            wgt = torch.softmax(sim * 10.0, dim=1)
            by = (wgt * wy).sum(1)
            bx = (wgt * wx).sum(1)

        px = bx / max(W1 - 1, 1) * max(nw1 - 1, 1) * sx1
        py = by / max(H1 - 1, 1) * max(nh1 - 1, 1) * sy1
        return torch.stack([px, py], -1).cpu().numpy()


class MASt3RFastGrid:
    """The group's pose protocol: a 60x40 query grid (2,400 points) through warp_at."""

    def __init__(self, ckpt=CKPT, device="cuda", **kw):
        self.dense = MASt3RFastDense(ckpt, device, **kw)
        self.model = self.dense.model          # for bench_complexity.modules_in()

    def __call__(self, i0, i1):
        h, w = i0.shape[:2]
        xs, ys = np.meshgrid(np.linspace(20, w - 20, 60), np.linspace(20, h - 20, 40))
        p0 = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32)
        return p0, self.dense.warp_at(i0, i1, p0)


def build_matcher(a, device="cuda"):
    """`--matcher mast3r_fast`. Baseline (non-fine-tuned) weights unless --extra_ckpt says otherwise.

    Either checkpoint works: the fp16 `..._fast.pth` file, or the ordinary fp32 one -- the
    fast path is applied by this wrapper, not carried by the weights. Env overrides
    MAST3R_PREP / MAST3R_AMP let the configuration be A/B'd without editing code.
    """
    ckpt = getattr(a, "extra_ckpt", None) or CKPT
    m = MASt3RFastGrid(ckpt, device)

    def f(i0, i1):          # closure: bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or "mast3r_base_fast")
