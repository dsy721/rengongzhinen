# 参数高效的鲁棒机器遗忘：LoRA + SAM (在 TOFU 上的实践)

基于 `Qwen2.5-1.5B-Instruct` 与 TOFU 数据集,复现 NPO / GradDiff 大模型遗忘方法、构建重学习攻击评估,并提出创新点 **LoRA + SAM 参数高效鲁棒遗忘**：仅训练 **0.59%** 的参数,即可获得**比全参方法更强的抗重学习攻击鲁棒性**。

## 核心结论

| 方法 | 可训练参数 | 干净 forget 概率 | 攻击(ep3)后 forget 概率 |
|---|---|---|---|
| 全参 NPO | 100% | 0.019 | 0.320 |
| 全参 NPO+SAM | 100% | 0.015 | 0.290 |
| LoRA NPO（无 SAM） | 0.59% | 0.058 | 0.405 |
| **LoRA NPO+SAM（本文）** | **0.59%** | 0.020 | **0.060** |

> forget 概率越低 = 遗忘越彻底 / 越抗攻击。在最强攻击(40 样本 × 3 epoch)下,LoRA NPO 被完全攻回原模型水平(0.751),而 **LoRA+SAM 仍稳定在 0.064**,几乎免疫,且模型通用能力(Model Utility)保持最高。

**三点结论**：(A) SAM 提升遗忘鲁棒性；(B) LoRA+SAM 用 0.59% 参数即达到甚至超过全参 SAM 的鲁棒性；(C) 鲁棒性增益来自 SAM——同为 LoRA,去掉 SAM 即在攻击下完全失守。

## 数据集与模型
- **数据集**：[TOFU](https://huggingface.co/datasets/locuslab/TOFU)(虚构作者问答；`forget01` 遗忘集 / `retain99` 保留集 / `*_perturbed` 用于 Truth Ratio)。
- **模型**：`Qwen/Qwen2.5-1.5B-Instruct`。
- **方法**：NPO(β=0.1)、GradDiff；SAM(ρ=0.01,扰动只作用于 forget 损失,LoRA 下只作用于 adapter)。

## 方法关键公式
- **NPO**：L = (2/β)·E[ softplus( β·(logπ_θ − logπ_ref) ) ],logπ 为 answer token 对数概率之**和**。
- **SAM**：δ = ρ·∇ℓ_f /‖∇ℓ_f‖₂,在 θ+δ 处算 forget 梯度,retain 损失不加扰动。
- **LoRA+SAM**：SAM 的扰动与更新只作用在 LoRA adapter 参数上。

## 环境
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 首次运行会自动下载模型与数据；缓存完成后可加 export HF_HUB_OFFLINE=1 加速
```
硬件：单卡 V100 32GB(无 bf16,使用 fp16 AMP)。

## 复现步骤

### 1. 训练起点模型 target(记住 forget01)
```bash
python -m src.finetune --split full --output_dir checkpoints/target
```

### 2. 生成四类遗忘模型
```bash
INIT=checkpoints/target

# 全参 NPO
python -m src.unlearn --method npo --tuning full --sam false \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt $INIT --output_dir checkpoints/unlearn_npo

# 全参 NPO+SAM
python -m src.unlearn --method npo --tuning full --sam true --rho 0.01 \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt $INIT --output_dir checkpoints/unlearn_npo_sam

# LoRA NPO（无 SAM）
python -m src.unlearn --method npo --tuning lora --sam false --lr 2e-4 --epochs 8 \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt $INIT --output_dir checkpoints/unlearn_npo_lora

# LoRA NPO+SAM（创新点）
python -m src.unlearn --method npo --tuning lora --sam true --rho 0.01 --lr 2e-4 --epochs 8 \
  --forget_split forget01 --retain_split retain99 \
  --init_ckpt $INIT --output_dir checkpoints/unlearn_npo_sam_lora
```

### 3. 评估(可选,复现指标)
```bash
# 参考模型 retain_ref（只在 retain99 上训练，从未见过 forget）
python -m src.finetune --split retain99 --output_dir checkpoints/retain_ref
python -m src.evaluate --model_path checkpoints/retain_ref \
  --forget_split forget01_perturbed --out checkpoints/eval_retain_ref.json --eval_batch_size 16

# 评估某个遗忘模型（forget 概率 / RougeL / Forget Quality / Model Utility）
python -m src.evaluate --model_path checkpoints/unlearn_npo_sam_lora \
  --forget_split forget01_perturbed --out checkpoints/eval_npo_sam_lora.json \
  --forget_quality_ref checkpoints/eval_retain_ref.json --eval_batch_size 16
```

### 4. 重学习攻击(验证鲁棒性)
```bash
python -m src.attack --model_path checkpoints/unlearn_npo_sam_lora \
  --forget_split forget01 --num_samples 20 --epochs 3 --output_dir checkpoints/_atk
python -m src.evaluate --model_path checkpoints/_atk \
  --forget_split forget01_perturbed --out checkpoints/eval_attack_npo_sam_lora.json \
  --forget_quality_ref checkpoints/eval_retain_ref.json --eval_batch_size 16
rm -rf checkpoints/_atk
```

### 5. 画图制表
```bash
python report/make_plots.py   # 生成 report/fig_forget_prob.png 等
```

## 四类模型超参对照
| 模型 | tuning | SAM | 学习率 | epochs |
|---|---|---|---|---|
| unlearn_npo | full | 否 | 1e-5 | 5 |
| unlearn_npo_sam | full | 是(ρ=0.01) | 1e-5 | 5 |
| unlearn_npo_lora | lora | 否 | 2e-4 | 8 |
| unlearn_npo_sam_lora | lora | 是(ρ=0.01) | 2e-4 | 8 |

## 目录结构
- `src/`：`finetune.py`(微调) / `unlearn.py`(遗忘入口) / `losses.py`(NPO·GradDiff·SAM 损失) / `attack.py`(重学习攻击) / `evaluate.py`(TOFU 指标) / `model.py` / `data.py`
- `configs/base.yaml`：默认超参；`report/make_plots.py`：画图；`checkpoints/*.json`：评估结果

## 说明
仓库不含模型权重(已 `.gitignore`)；按上述步骤可从头复现。