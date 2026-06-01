import os
import sys
import json
import time
import argparse
import logging

import yaml
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import load_model_and_tokenizer
from src.data import build_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("attack")

DEVICE = "cuda"


def make_collator(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            ids = list(b["input_ids"])
            lbl = list(b["labels"])
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lbl + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="unlearned checkpoint to attack (e.g. checkpoints/unlearn_npo)")
    ap.add_argument("--forget_split", default="forget01",
                    help="forget config used as the attacker's relearning data")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=None,
                    help="attacker budget: how many forget examples to relearn on (None=all)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    max_length = cfg.get("max_length", 128)

    if args.smoke:
        args.epochs = 1
        args.num_samples = 4

    # Load the unlearned model with fp32 weights for stable training (fp16 only as AMP).
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, bf16=False, fp16=True, weight_dtype="fp32",
    )
    model.to(DEVICE)
    model.train()
    # Relearning attack must update ALL parameters.
    for p in model.parameters():
        p.requires_grad_(True)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    ds = build_dataset(args.forget_split, tokenizer, max_length)
    if args.num_samples is not None and args.num_samples < len(ds):
        ds = ds.select(range(args.num_samples))
    logger.info("attack relearning on %d forget examples, %d epochs",
                len(ds), args.epochs)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collator(pad_id))

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
                loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % 5 == 0:
                logger.info("epoch %d step %d | loss %.4f", epoch, step, float(loss.item()))
            step += 1

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
    info = {
        "attacked_from": args.model_path,
        "forget_split": args.forget_split,
        "epochs": args.epochs,
        "num_samples": len(ds),
        "lr": args.lr,
        "time_sec": round(time.time() - t0, 1),
        "peak_mem_MB": int(peak),
    }
    with open(os.path.join(args.output_dir, "attack_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info("attack done -> %s | time=%.1fs peak_mem=%dMB",
                args.output_dir, info["time_sec"], info["peak_mem_MB"])


if __name__ == "__main__":
    main()
