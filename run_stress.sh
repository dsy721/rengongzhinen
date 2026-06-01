#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EB=16
# 两种更强攻击：(a) 20样本×5epoch  (b) 40样本×3epoch
for SRC in unlearn_npo_lora unlearn_npo_sam_lora; do
  echo "###### ${SRC}: 20样本 x 5epoch ######"
  rm -rf checkpoints/_tmp_attack
  python -m src.attack --model_path checkpoints/${SRC} --forget_split forget01 \
    --num_samples 20 --epochs 5 --output_dir checkpoints/_tmp_attack
  python -m src.evaluate --model_path checkpoints/_tmp_attack --forget_split forget01_perturbed \
    --out checkpoints/eval_attack_${SRC}_ep5.json \
    --forget_quality_ref checkpoints/eval_retain_ref.json --eval_batch_size $EB
  rm -rf checkpoints/_tmp_attack

  echo "###### ${SRC}: 40样本 x 3epoch ######"
  rm -rf checkpoints/_tmp_attack
  python -m src.attack --model_path checkpoints/${SRC} --forget_split forget01 \
    --num_samples 40 --epochs 3 --output_dir checkpoints/_tmp_attack
  python -m src.evaluate --model_path checkpoints/_tmp_attack --forget_split forget01_perturbed \
    --out checkpoints/eval_attack_${SRC}_n40ep3.json \
    --forget_quality_ref checkpoints/eval_retain_ref.json --eval_batch_size $EB
  rm -rf checkpoints/_tmp_attack
done
echo "ALL DONE stress"
