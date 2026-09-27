#!/bin/bash
# Set-up for the version comparison, run ONCE on the login node (it needs internet):
#   bash ~/projects/eval/jobs/setup_versions.sh
# Clones the official repositories into ~/projects, builds two new uv envs (LoMa and RoMa v2 both
# need torch 2.8, which no existing env has), pre-fetches every weight file into the torch hub cache
# the jobs use (compute nodes have no internet), and prints commits + md5 for the Versions table.
set -e
UV=~/.local/bin/uv
P=~/projects
HUB=/cta/scratch/h-abdelaziz/torch_cache/hub
mkdir -p $HUB/checkpoints /cta/scratch/h-abdelaziz/weights/LoFTR
export TORCH_HOME=/cta/scratch/h-abdelaziz/torch_cache

echo "== 1. LoMa (github.com/davnords/LoMa)"
[ -d $P/LoMa ] || git clone https://github.com/davnords/LoMa.git $P/LoMa
[ -d ~/loma_env ] || $UV venv ~/loma_env --python 3.10
$UV pip install --python ~/loma_env/bin/python torch==2.8.0 torchvision==0.23.0
$UV pip install --python ~/loma_env/bin/python -e $P/LoMa
# LoMa depends on opencv-python, whose cv2 needs libGL (absent on the login node); the headless build is the same module without it
$UV pip uninstall --python ~/loma_env/bin/python opencv-python >/dev/null 2>&1 || true
$UV pip install --reinstall --python ~/loma_env/bin/python opencv-python-headless
~/loma_env/bin/python -c "import torch, cv2, importlib.util as u; assert u.find_spec('loma'); print('   loma installed, cv2', cv2.__version__, 'torch', torch.__version__)"

echo "== 2. RoMa v2 (github.com/Parskatt/RoMaV2)"
[ -d $P/RoMaV2 ] || git clone https://github.com/Parskatt/RoMaV2.git $P/RoMaV2
[ -d ~/romav2_env ] || $UV venv ~/romav2_env --python 3.12
$UV pip install --python ~/romav2_env/bin/python torch==2.8.0 torchvision==0.23.0 fused-local-corr==0.3.28
$UV pip install --python ~/romav2_env/bin/python -e $P/RoMaV2
$UV pip install --reinstall --python ~/romav2_env/bin/python opencv-python-headless
~/romav2_env/bin/python -c "import torch, cv2, importlib.util as u; assert u.find_spec('romav2'); print('   romav2 installed, cv2', cv2.__version__, 'torch', torch.__version__)"
echo "   fetching the DINOv3 code the repo pulls through torch.hub (pinned commit) into $HUB"
~/romav2_env/bin/python -c "import torch; torch.hub.load('facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa', 'dinov3_vitl16', pretrained=False, weights=None, skip_validation=True, trust_repo=True); print('   dinov3 code cached')"

echo "== 3. LoFTR (github.com/zju3dv/LoFTR), runs in the existing ~/eloftr_env"
[ -d $P/LoFTR ] || git clone https://github.com/zju3dv/LoFTR.git $P/LoFTR
if [ ! -f /cta/scratch/h-abdelaziz/weights/LoFTR/weights/outdoor_ds.ckpt ]; then
  echo "   downloading the README's Google Drive folder (indoor_ds, indoor_ds_new, outdoor_ds; ~45 MB each)"
  $UV tool run gdown --folder https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf -O /cta/scratch/h-abdelaziz/weights/LoFTR
fi
~/eloftr_env/bin/python -c "import yacs, einops, kornia; print('   eloftr_env has yacs/einops/kornia')" || $UV pip install --python ~/eloftr_env/bin/python yacs

echo "== 4. weights into the torch hub cache (skipped if present and complete)"
fetch() {  # url expected_min_MB
  f=$HUB/checkpoints/$(basename $1)
  if [ -f "$f" ] && [ $(du -m "$f" | cut -f1) -ge $2 ]; then echo "   have $(basename $1)"; return; fi
  echo "   $(basename $1)"; curl -L --retry 3 -o "$f" "$1"
}
fetch https://github.com/davnords/storage/releases/download/loma/loma_B.pt 700
fetch https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth 20
fetch https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth 60
fetch https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt 1000
ls -la $HUB/checkpoints/dinov2_vitl14_pretrain.pth >/dev/null && echo "   have dinov2_vitl14_pretrain.pth"

echo "== 5. versions (also computed by collect_results.py)"
for r in LoMa RoMaV2 LoFTR EfficientLoFTR RoMa; do printf '   %-15s %s\n' $r "$(git -C $P/$r log -1 --format='%h %cs %s' | cut -c1-70)"; done
md5sum $HUB/checkpoints/loma_B.pt $HUB/checkpoints/romav2.0.1.pt /cta/scratch/h-abdelaziz/weights/LoFTR/weights/outdoor_ds.ckpt | sed 's/^/   /'
echo "== done. Next: sbatch jobs/loma_smoke.sbatch jobs/romav2_smoke.sbatch jobs/eloftr_versions_smoke.sbatch"
