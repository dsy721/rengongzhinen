# TODO: 反学习流程模块，占位后续实现 TOFU 阶段 3A 的手写训练入口。

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse, json, itertools, yaml, torch
from torch.utils.data import DataLoader
import bitsandbytes as bnb
from src.utils import set_seed, get_logger, Timer, gpu_mem_mb
from src.data import build_dataset
from src.model import load_model_and_tokenizer, wrap_lora, count_trainable_params
from src.losses import grad_diff_forget_loss, npo_forget_loss, retain_ce_loss

logger = get_logger("unlearn")
DEVICE = "cuda"


def make_collator(pad_token_id):
    def collate(features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"]); am = list(f["attention_mask"]); lb = list(f["labels"])
            padlen = maxlen - len(ids)
            input_ids.append(ids + [pad_token_id] * padlen)
            attention_mask.append(am + [0] * padlen)
            labels.append(lb + [-100] * padlen)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def compute_forget_loss(model, ref_model, fb, args):
    if args.method == "graddiff":
        return grad_diff_forget_loss(model, fb)
    return npo_forget_loss(model, ref_model, fb, beta=args.beta)


def sam_step(model, ref_model, fb, rb, optimizer, args, lam):
    """SAM 鲁棒遗忘一步（Fan et al. ICML2025 Alg.A1），纯 fp32 运行。
       delta = rho * grad_f / ||grad_f||_2 ; gf = grad_f(theta+delta) ;
       gr = grad_r(theta) ; theta -= eta * (gf + lam * gr)."""
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad(set_to_none=True)
    forget_loss = compute_forget_loss(model, ref_model, fb, args)
    forget_loss.backward()

    with torch.no_grad():
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in params if p.grad is not None]), 2
        )
        scale = args.rho / (grad_norm + 1e-12)
        e_ws = []
        for p in params:
            if p.grad is None:
                e_ws.append(None)
            else:
                e_w = p.grad * scale
                p.add_(e_w)
                e_ws.append(e_w)

    optimizer.zero_grad(set_to_none=True)
    forget_loss2 = compute_forget_loss(model, ref_model, fb, args)
    forget_loss2.backward()

    with torch.no_grad():
        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.sub_(e_w)

    retain_loss = retain_ce_loss(model, rb)
    (lam * retain_loss).backward()
    optimizer.step()

    L = forget_loss2.detach() + lam * retain_loss.detach()
    return forget_loss2.detach(), retain_loss.detach(), L


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["graddiff", "npo"], required=True)
    p.add_argument("--tuning", choices=["full", "lora"], required=True)
    p.add_argument("--sam", choices=["true", "false"], default="false")
    p.add_argument("--forget_split", default="forget01")
    p.add_argument("--retain_split", default="retain99")
    p.add_argument("--init_ckpt", default="checkpoints/target")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lambda_retain", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    lam = args.lambda_retain if args.lambda_retain is not None else cfg["lambda_retain"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    exp_name = os.path.basename(os.path.normpath(args.output_dir))

    model, tok = load_model_and_tokenizer(args.init_ckpt, bf16=False, fp16=False, weight_dtype="fp32")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(DEVICE)
    if args.tuning == "lora":
        model = wrap_lora(model)
    model.train()

    ref_model = None
    if args.method == "npo":
        ref_model, _ = load_model_and_tokenizer(args.init_ckpt, bf16=False, fp16=False, weight_dtype="fp16")
        ref_model.to(DEVICE).eval()
        for p_ in ref_model.parameters():
            p_.requires_grad = False

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    collate = make_collator(pad_id)
    f_ds = build_dataset(args.forget_split, tok, cfg["max_length"])
    r_ds = build_dataset(args.retain_split, tok, cfg["max_length"])
    if args.smoke:
        f_ds = f_ds.select(range(min(16, len(f_ds))))
        r_ds = r_ds.select(range(min(16, len(r_ds))))
    bs = cfg["batch_size"]
    f_loader = DataLoader(f_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    r_loader = DataLoader(r_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    r_iter = itertools.cycle(r_loader)

    opt_cls = getattr(bnb.optim, "AdamW8bit", None)
    optimizer = opt_cls([p for p in model.parameters() if p.requires_grad], lr=lr) if opt_cls else torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") else torch.cuda.amp.GradScaler()
    epochs = 1 if args.smoke else (args.epochs if args.epochs is not None else cfg["epochs"])

    trn, tot, ratio = count_trainable_params(model)
    logger.info(f"trainable params: {trn} / {tot} ({ratio:.4%})")
    logger.info(f"lr={lr} epochs={epochs} tuning={args.tuning} sam={args.sam}")

    step = 0
    with Timer() as timer:
        for ep in range(epochs):
            for fb in f_loader:
                fb = to_device(fb); rb = to_device(next(r_iter))
                if args.sam == "true":
                    forget_loss, retain_loss, L = sam_step(model, ref_model, fb, rb, optimizer, args, lam)
                else:
                    optimizer.zero_grad()
                    with torch.autocast("cuda", dtype=torch.float16):
                        if args.method == "graddiff":
                            forget_loss = grad_diff_forget_loss(model, fb)
                        else:
                            forget_loss = npo_forget_loss(model, ref_model, fb, beta=args.beta)
                        retain_loss = retain_ce_loss(model, rb)
                        L = forget_loss + lam * retain_loss
                    scaler.scale(L).backward()
                    scaler.step(optimizer); scaler.update()
                if step % 10 == 0:
                    logger.info(f"step {step} | forget {forget_loss.item():.4f} | retain {retain_loss.item():.4f} | L {L.item():.4f}")
                step += 1

    os.makedirs(f"results/{exp_name}", exist_ok=True)
    json.dump({"exp_name": exp_name, "method": args.method, "tuning": args.tuning,
               "sam": args.sam, "train_time_sec": timer.elapsed, "peak_mem_mb": gpu_mem_mb(),
               "trainable_params": trn, "total_params": tot, "trainable_ratio": ratio},
              open(f"results/{exp_name}/cost.json", "w"), ensure_ascii=False, indent=2)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.tuning == "lora" and hasattr(model, "merge_and_unload"):
        logger.info("merging LoRA adapter into base weights before saving full model")
        model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    logger.info(f"saved to {args.output_dir}, time={timer.elapsed:.1f}s, peak_mem={gpu_mem_mb():.0f}MB")


if __name__ == "__main__":
    main()
