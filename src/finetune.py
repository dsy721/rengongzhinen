# TODO: 微调入口，使用 HuggingFace Trainer 完成 TOFU 的标准 SFT 阶段训练。

from __future__ import annotations

import argparse
import os
import yaml

import src  # noqa: F401

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments

from src.data import build_dataset
from src.model import load_model_and_tokenizer
from src.utils import Timer, get_logger, gpu_mem_mb, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg["epochs"] = 1

    logger = get_logger("finetune")
    set_seed(cfg["seed"])
    model, tokenizer = load_model_and_tokenizer(
        cfg["model_name"],
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        weight_dtype="fp32",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(args.split, tokenizer, cfg["max_length"])
    if args.smoke:
        dataset = dataset.select(range(min(20, len(dataset))))

    def collate_fn(features):
        filtered = []
        for f in features:
            filtered.append(
                {
                    "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
                    "labels": torch.tensor(f["labels"], dtype=torch.long),
                    "attention_mask": torch.tensor(f["attention_mask"], dtype=torch.long),
                }
            )
        return {
            "input_ids": pad_sequence([f["input_ids"] for f in filtered], batch_first=True, padding_value=tokenizer.pad_token_id),
            "labels": pad_sequence([f["labels"] for f in filtered], batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence([f["attention_mask"] for f in filtered], batch_first=True, padding_value=0),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="no" if args.smoke else "epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    with Timer() as t:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    logger.info("Training finished in %.2f sec", t.elapsed)
    logger.info("GPU peak memory: %.2f MB", gpu_mem_mb())


if __name__ == "__main__":
    main()
