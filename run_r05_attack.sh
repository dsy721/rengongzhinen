#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/2] relearning attack (20 samples) on rho=0.05 model ..."
python -m src.attack --model_path checkpoints/unlearn_npo_sam_r05 \
  --forget_split forget01 --num_samples 20 --output_dir checkpoints/attack_npo_sam_r05

echo "[2/2] eval after attack ..."
python -m src.evaluate --model_path checkpoints/attack_npo_sam_r05 \
  --forget_split forget01_perturbed --out checkpoints/eval_attack_npo_sam_r05.json \
  --forget_quality_ref checkpoints/eval_retain_ref.json

echo "cleanup: rho=0.05 checkpoints no longer needed, keep only json ..."
rm -rf checkpoints/unlearn_npo_sam_r05 checkpoints/attack_npo_sam_r05
echo "ALL DONE r05 attack"
