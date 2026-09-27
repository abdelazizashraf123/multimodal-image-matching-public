"""Matcher registry shared by every evaluator and by the complexity scripts.

build_matcher(a, device) -> (match, tag)
    match(img0_bgr, img1_bgr) -> (kpts0, kpts1) float pixel coordinates in the frame of the
    images given; tag names the arm in result paths. `a` is an argparse Namespace with at least
    .matcher; per-matcher extras: roma -> .pretrained .dinov2 .finetuned; eloftr / eloftr_opt /
    loftr / mast3r / mast3r_nn -> .extra_ckpt (checkpoint); romav2 -> .romav2_setting;
    loma -> .loma_variant.

Arms:
    roma        original RoMa (v1), roma_outdoor.pth                       venv ~/myenv
    romav2      RoMa v2, default "precise" setting                          venv ~/romav2_env
    loftr       original LoFTR, outdoor_ds.ckpt (ELoFTR's predecessor)      venv ~/eloftr_env
    eloftr      EfficientLoFTR, full config, fp32                           venv ~/eloftr_env
    eloftr_opt  EfficientLoFTR, opt config + fp16 (same weights)            venv ~/eloftr_env
    mast3r      MASt3R, group dense-grid protocol; mast3r_nn = native NN    venv ~/mast3r_env
    loma        LoMa-B (complexity row only)                                venv ~/loma_env
Imports are lazy on purpose: the repositories pin incompatible packages and shadow each other's
top-level package names (`src`, `romatch`). One matcher per process.
"""
KNOWN = ("roma", "romav2", "loftr", "eloftr", "eloftr_opt", "mast3r", "mast3r_nn", "mast3r_fast", "loma", "splg")


def build_matcher(a, device="cuda"):
    name = getattr(a, "matcher", None) or "roma"
    if name == "roma":
        import roma_matcher
        return roma_matcher.build_matcher(a, device)
    if name == "romav2":
        from matchers.romav2 import build_matcher as build
        return build(a, device)
    if name == "loftr":
        from matchers.loftr import build_matcher as build
        return build(a, device)
    if name == "eloftr":
        from matchers.eloftr import build_matcher as build
        return build(a, device, mode="full")
    if name == "eloftr_opt":
        from matchers.eloftr import build_matcher as build
        return build(a, device, mode="opt")
    if name == "mast3r":
        from matchers.mast3r import build_matcher as build
        return build(a, device, mode="grid")
    if name == "mast3r_nn":
        from matchers.mast3r import build_matcher as build
        return build(a, device, mode="nn")
    if name == "mast3r_fast":                # a-shahzeen's fast path (her code copy), grid protocol, mast3r_env
        from matchers.mast3r_fast import build_matcher as build
        return build(a, device)
    if name == "loma":
        from matchers.loma import build_matcher as build
        return build(a, device)
    if name == "splg":                       # MINIMA-LightGlue (SuperPoint + LightGlue, MINIMA weights), minima_env
        from matchers.splg import build_matcher as build
        return build(a, device)
    raise ValueError(f"unknown matcher {name!r}; known: {KNOWN}")
