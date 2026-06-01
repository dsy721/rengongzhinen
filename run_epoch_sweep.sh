#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SRC in unlearn_npo unlearn_npo_sam; do
  for EP in 1 2 3; do
    echo "============== attack ${SRC} epochs=${EP} =============="
    rm -rf checkpoints/_tmp_attack
    python -m src.attack --model_path checkpoints/${SRC} \
      --forget_split forget01 --num_samples 20 --epochs ${EP} \
      --output_dir checkpoints/_tmp_attack
    python -m src.evaluate --model_path checkpoints/_tmp_attack \
      --forget_split forget01_perturbed \
      --out checkpoints/eval_attack_${SRC}_ep${EP}.json \
      --forget_quality_ref checkpoints/eval_retain_ref.json
    rm -rf checkpoints/_tmp_attack
    echo "---- done ${SRC} ep${EP}, json saved ----"
  done
done
echo "ALL DONE epoch sweep"
