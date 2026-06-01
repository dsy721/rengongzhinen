#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EB=16

echo "===== [1/4] retrain NPO+LoRA (merged save) lr2e-4 ep8 ====="
python -m src.unlearn --method npo --tuning lora --sam false --lr 2e-4 --epochs 8 \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt checkpoints/target --output_dir checkpoints/unlearn_npo_lora

echo "===== [2/4] eval NPO+LoRA clean ====="
python -m src.evaluate --model_path checkpoints/unlearn_npo_lora \
  --forget_split forget01_perturbed --out checkpoints/eval_npo_lora.json \
  --forget_quality_ref checkpoints/eval_retain_ref.json --eval_batch_size $EB

echo "===== [3/4] retrain NPO+SAM+LoRA (merged save) lr2e-4 ep8 rho0.01 ====="
python -m src.unlearn --method npo --tuning lora --sam true --rho 0.01 --lr 2e-4 --epochs 8 \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt checkpoints/target --output_dir checkpoints/unlearn_npo_sam_lora

echo "===== [4/4] eval NPO+SAM+LoRA clean ====="
python -m src.evaluate --model_path checkpoints/unlearn_npo_sam_lora \
  --forget_split forget01_perturbed --out checkpoints/eval_npo_sam_lora.json \
  --forget_quality_ref checkpoints/eval_retain_ref.json --eval_batch_size $EB

echo "===== [5] attack sweep ep1/2/3 for both LoRA models ====="
for SRC in unlearn_npo_lora unlearn_npo_sam_lora; do
  for EP in 1 2 3; do
    echo "------ attack ${SRC} ep${EP} ------"
    rm -rf checkpoints/_tmp_attack
    python -m src.attack --model_path checkpoints/${SRC} \
      --forget_split forget01 --num_samples 20 --epochs ${EP} \
      --output_dir checkpoints/_tmp_attack
    python -m src.evaluate --model_path checkpoints/_tmp_attack \
      --forget_split forget01_perturbed \
      --out checkpoints/eval_attack_${SRC}_ep${EP}.json \
      --forget_quality_ref checkpoints/eval_retain_ref.json --eval_batch_size $EB
    rm -rf checkpoints/_tmp_attack
  done
done
echo "ALL DONE lora fixed"
