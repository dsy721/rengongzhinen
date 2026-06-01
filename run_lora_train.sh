#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== [1/4] train NPO+LoRA (no SAM) lr=2e-4 epochs=8 ====="
python -m src.unlearn --method npo --tuning lora --sam false --lr 2e-4 --epochs 8 \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt checkpoints/target --output_dir checkpoints/unlearn_npo_lora

echo "===== [2/4] eval NPO+LoRA (clean) ====="
python -m src.evaluate --model_path checkpoints/unlearn_npo_lora \
  --forget_split forget01_perturbed --out checkpoints/eval_npo_lora.json \
  --forget_quality_ref checkpoints/eval_retain_ref.json

echo "===== [3/4] train NPO+SAM+LoRA lr=2e-4 epochs=8 rho=0.01 ====="
python -m src.unlearn --method npo --tuning lora --sam true --rho 0.01 --lr 2e-4 --epochs 8 \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt checkpoints/target --output_dir checkpoints/unlearn_npo_sam_lora

echo "===== [4/4] eval NPO+SAM+LoRA (clean) ====="
python -m src.evaluate --model_path checkpoints/unlearn_npo_sam_lora \
  --forget_split forget01_perturbed --out checkpoints/eval_npo_sam_lora.json \
  --forget_quality_ref checkpoints/eval_retain_ref.json

echo "ALL DONE lora retrain"
