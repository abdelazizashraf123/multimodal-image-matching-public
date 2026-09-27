"""MASt3R (MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) as a matcher, two protocols.

'mast3r'    : a-shahzeen's dense-query protocol (mast3r_dense.py, verbatim below): 512 px long edge,
              the per-pixel descriptor maps are queried at a 60x40 grid of 2,400 points in image 0
              (20 px margin), each query's descriptor is correlated against every position of
              image 1, argmax, then a softmax centroid over a 5x5 window for sub-grid accuracy.
              This produced the group's MASt3R rows (RSMD 18.2/23.2/27.7, complexity 688.6 M /
              163 ms / 2400 matches), so RSMD, DSERT and complexity use it to stay comparable.
'mast3r_nn' : MASt3R's own sparse matching, fast_reciprocal_NNs(subsample=8, dot), as in the
              repository README and in a-shahzeen's MINIMA-protocol scripts; a labelled extra.
Resolution handicap (her note): 512 px long edge makes the descriptor grid ~2.5x coarser than
a 1280 px image, so tight thresholds are structurally penalised; read @10/@20 beside RoMa.
Code and weights are the user's own: ~/projects/mast3r (official, with dust3r) and
/cta/scratch/h-abdelaziz/weights/mast3r/... (md5 42f8ec7d2a413aa214ebd66c69769ecc).
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as tvf

MAST3R = "/cta/users/h-abdelaziz/projects/mast3r"
CKPT = "/cta/scratch/h-abdelaziz/weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _import_mast3r():
    if MAST3R not in sys.path:
        sys.path.insert(0, MAST3R)
    import mast3r.utils.path_to_dust3r  # noqa: F401  (puts dust3r on the path)
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    return AsymmetricMASt3R, inference


class MASt3RDense:
    """Exposes .warp_at(img0_bgr, img1_bgr, pts0) -> pts1 (a-shahzeen's mast3r_dense.py)."""

    def __init__(self, ckpt_path, device="cuda", size=512, win=2):
        AsymmetricMASt3R, self._inference = _import_mast3r()
        self.device = device
        self.size = size
        self.win = win          # half-width of the sub-grid refinement window
        self.model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device).eval()

    def _prep(self, img_bgr, idx):
        h, w = img_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        long_edge = max(w, h)
        nw = max(16, (int(round(w * self.size / long_edge)) // 16) * 16)
        nh = max(16, (int(round(h * self.size / long_edge)) // 16) * 16)
        t = ImgNorm(pil.resize((nw, nh), Image.LANCZOS))[None]
        view = dict(img=t, true_shape=np.int32([[nh, nw]]), idx=idx, instance=str(idx))
        return view, (w / nw, h / nh), (nw, nh)

    @torch.no_grad()
    def warp_at(self, img0_bgr, img1_bgr, pts0):
        v0, (sx0, sy0), (nw0, nh0) = self._prep(img0_bgr, 0)
        v1, (sx1, sy1), (nw1, nh1) = self._prep(img1_bgr, 1)
        out = self._inference([(v0, v1)], self.model, self.device, batch_size=1, verbose=False)
        # dust3r's inference() returns predictions on CPU regardless of the model device
        d0 = out["pred1"]["desc"].squeeze(0).detach().float().to(self.device)   # (H0,W0,D)
        d1 = out["pred2"]["desc"].squeeze(0).detach().float().to(self.device)   # (H1,W1,D)
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


class MASt3RGrid:
    """a-shahzeen's pose protocol: 60x40 query grid (2,400 points) through warp_at."""

    def __init__(self, ckpt=CKPT, device="cuda"):
        self.dense = MASt3RDense(ckpt, device)
        self.model = self.dense.model          # for bench_complexity.modules_in()

    def __call__(self, i0, i1):
        h, w = i0.shape[:2]
        xs, ys = np.meshgrid(np.linspace(20, w - 20, 60), np.linspace(20, h - 20, 40))
        p0 = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32)
        return p0, self.dense.warp_at(i0, i1, p0)


class MASt3RNN:
    """MASt3R's official sparse matching (README / a-shahzeen's MINIMA-protocol scripts)."""

    def __init__(self, ckpt=CKPT, device="cuda"):
        AsymmetricMASt3R, self._inference = _import_mast3r()
        from mast3r.fast_nn import fast_reciprocal_NNs
        self._fast_nn = fast_reciprocal_NNs
        self.device = device
        self.size = 512
        self.model = AsymmetricMASt3R.from_pretrained(ckpt).to(device).eval()

    def _prep(self, img_bgr, idx):
        h, w = img_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        long_edge = max(w, h)
        new_w = max(16, (int(round(w * self.size / long_edge)) // 16) * 16)
        new_h = max(16, (int(round(h * self.size / long_edge)) // 16) * 16)
        tensor = ImgNorm(pil.resize((new_w, new_h), Image.LANCZOS))[None]
        view = dict(img=tensor, true_shape=np.int32([[new_h, new_w]]), idx=idx, instance=str(idx))
        return view, (w / new_w, h / new_h)

    @torch.no_grad()
    def __call__(self, img0_bgr, img1_bgr):
        view0, (sx0, sy0) = self._prep(img0_bgr, 0)
        view1, (sx1, sy1) = self._prep(img1_bgr, 1)
        output = self._inference([(view0, view1)], self.model, self.device, batch_size=1, verbose=False)
        v1, v2 = output["view1"], output["view2"]
        desc1 = output["pred1"]["desc"].squeeze(0).detach()
        desc2 = output["pred2"]["desc"].squeeze(0).detach()
        m0, m1 = self._fast_nn(desc1, desc2, subsample_or_initxy1=8, device=self.device,
                               dist="dot", block_size=2 ** 13)
        H0, W0 = v1["true_shape"][0]
        H1, W1 = v2["true_shape"][0]
        valid = ((m0[:, 0] >= 3) & (m0[:, 0] < int(W0) - 3) & (m0[:, 1] >= 3) & (m0[:, 1] < int(H0) - 3) &
                 (m1[:, 0] >= 3) & (m1[:, 0] < int(W1) - 3) & (m1[:, 1] >= 3) & (m1[:, 1] < int(H1) - 3))
        m0, m1 = m0[valid], m1[valid]
        k0 = m0.astype(np.float32) * np.array([sx0, sy0], dtype=np.float32)
        k1 = m1.astype(np.float32) * np.array([sx1, sy1], dtype=np.float32)
        return k0, k1


def build_matcher(a, device="cuda", mode="grid"):
    ckpt = getattr(a, "extra_ckpt", None) or CKPT
    if mode == "grid":
        m = MASt3RGrid(ckpt, device)
        default_tag = "mast3r_base"          # a-shahzeen's tag for the same protocol
    else:
        m = MASt3RNN(ckpt, device)
        default_tag = "mast3r_nn"

    def f(i0, i1):          # closure: complexity/bench_complexity.modules_in() digs the model out of it
        return m(i0, i1)

    return f, (getattr(a, "tag", None) or default_tag)
