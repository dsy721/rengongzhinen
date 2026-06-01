import os
import sys
import json
import math
import argparse
import logging
from collections import defaultdict

import yaml
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import ks_2samp, hmean
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import load_model_and_tokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

DEVICE = "cuda"


# ----------------------------- tokenization helpers -----------------------------
def encode_qa(tokenizer, question, answer, max_length):
    """Build chat-formatted input_ids + labels, masking the question (prompt) with -100.
    Uses tokenize=False to get plain text, then tokenizes to a FLAT id list — this avoids
    the batched/nested ([[...]]) return shape of apply_chat_template(tokenize=True) in
    transformers 5.x. Returns (input_ids, labels) as python lists."""
    user_msgs = [{"role": "user", "content": question}]
    full_msgs = user_msgs + [{"role": "assistant", "content": answer}]
    prompt_text = tokenizer.apply_chat_template(
        user_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    input_ids = full_ids[:max_length]
    labels = list(input_ids)
    plen = min(len(prompt_ids), len(input_ids))
    for i in range(plen):
        labels[i] = -100
    return input_ids, labels


def _pad_token_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def batch_encode_qa(tokenizer, questions, answers, max_length):
    input_ids_list, labels_list = [], []
    for q, a in zip(questions, answers):
        input_ids, labels = encode_qa(tokenizer, q, a, max_length)
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return input_ids_list, labels_list


@torch.no_grad()
def batch_answer_nll(model, tokenizer, questions, answers, max_length):
    """Mean per-token NLL over answer tokens for a batch.
    Batch and single-sample results are equivalent when padding/masks are correct."""
    input_ids_list, labels_list = batch_encode_qa(tokenizer, questions, answers, max_length)
    if not input_ids_list:
        return []

    max_len = max(len(x) for x in input_ids_list)
    pad_id = _pad_token_id(tokenizer)
    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        labels[i, :L] = torch.tensor(lbl, dtype=torch.long)
        attn[i, :L] = 1
    input_ids = input_ids.to(DEVICE)
    labels = labels.to(DEVICE)
    attn = attn.to(DEVICE)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    safe_labels = shifted_labels.clone()
    safe_labels[~mask] = 0
    token_nll = torch.zeros_like(shifted_labels, dtype=torch.float32)
    gathered = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_nll[mask] = gathered[mask]
    per_sample = []
    for i in range(token_nll.size(0)):
        m = mask[i]
        if not torch.any(m):
            per_sample.append(None)
        else:
            per_sample.append(float(token_nll[i][m].mean().item()))
    return per_sample


@torch.no_grad()
def answer_nll(model, tokenizer, question, answer, max_length):
    """Mean per-token NLL over the answer tokens. Returns float (or None if no answer tokens)."""
    batch = batch_answer_nll(model, tokenizer, [question], [answer], max_length)
    return batch[0] if batch else None


def norm_prob(nll):
    """Length-normalized probability = exp(-mean_token_nll)."""
    if nll is None:
        return 0.0
    return math.exp(-nll)


# ----------------------------- generation + ROUGE -----------------------------
@torch.no_grad()
def generate_answer(model, tokenizer, question, max_new_tokens=64):
    return generate_answers(model, tokenizer, [question], max_new_tokens=max_new_tokens)[0]


@torch.no_grad()
def generate_answers(model, tokenizer, questions, max_new_tokens=64):
    prompts = [build_prompt_text(tokenizer, q) for q in questions]
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=_pad_token_id(tokenizer),
        )
    preds = []
    for i, plen in enumerate(prompt_lens):
        gen = out[i][plen:]
        preds.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return preds


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def rougeL_recall(pred, ref):
    """Word-level ROUGE-L recall = LCS(pred,ref) / len(ref). Approximation of rouge_score."""
    p = pred.lower().split()
    r = ref.lower().split()
    if len(r) == 0:
        return 0.0
    return _lcs_len(p, r) / len(r)


# ----------------------------- per-example metrics -----------------------------
def get_field(ex, *keys):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return None


def truth_ratio_for_example(model, tokenizer, ex, max_length):
    """R = exp(avg_perturb_nll - paraphrase_nll). Returns float or None."""
    q = ex["question"]
    para = get_field(ex, "paraphrased_answer", "answer")
    perts = get_field(ex, "perturbed_answer", "perturbed_answers")
    if para is None or perts is None:
        return None
    if isinstance(perts, str):
        perts = [perts]
    nlls = batch_answer_nll(model, tokenizer, [q] * (1 + len(perts)), [para] + list(perts), max_length)
    para_nll = nlls[0]
    pert_nlls = [n for n in nlls[1:] if n is not None]
    if para_nll is None or len(pert_nlls) == 0:
        return None
    avg_pert = float(np.mean(pert_nlls))
    return math.exp(avg_pert - para_nll)


def mcq_probability(model, tokenizer, ex, max_length):
    """P_true / (P_true + sum P_perturbed), all length-normalized."""
    q = ex["question"]
    perts = get_field(ex, "perturbed_answer", "perturbed_answers")
    if isinstance(perts, str):
        perts = [perts]
    nlls = batch_answer_nll(model, tokenizer, [q] * (1 + len(perts)), [ex["answer"]] + list(perts), max_length)
    p_true = norm_prob(nlls[0])
    p_perts = [norm_prob(n) for n in nlls[1:]]
    denom = p_true + sum(p_perts)
    if denom == 0:
        return 0.0
    return p_true / denom


# ----------------------------- dataset-level evaluation -----------------------------
def load_tofu_config(name, limit=None):
    ds = load_dataset("locuslab/TOFU", name=name, split="train")
    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))
    return ds


def build_prompt_text(tokenizer, question):
    user_msgs = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)


def eval_forget(model, tokenizer, forget_split, max_length, limit, eval_batch_size):
    """Returns dict with raw truth_ratio array, mean rougeL_recall, mean probability."""
    ds = load_tofu_config(forget_split, limit)
    trs, rouges, probs = [], [], []
    for i in tqdm(range(0, len(ds), eval_batch_size), desc="forget"):
        batch = ds[i:i + eval_batch_size]
        if isinstance(batch, dict):
            keys = list(batch.keys())
            batch = [dict(zip(keys, vals)) for vals in zip(*batch.values())]
        qs = [ex["question"] for ex in batch]
        gts = [ex["answer"] for ex in batch]
        preds = generate_answers(model, tokenizer, qs)
        for ex, q, gt, pred in zip(batch, qs, gts, preds):
            tr = truth_ratio_for_example(model, tokenizer, ex, max_length)
            if tr is not None:
                trs.append(tr)
            probs.append(norm_prob(answer_nll(model, tokenizer, q, gt, max_length)))
            rouges.append(rougeL_recall(pred, gt))
    return {
        "truth_ratio": trs,
        "rougeL_recall": float(np.mean(rouges)) if rouges else 0.0,
        "probability": float(np.mean(probs)) if probs else 0.0,
        "n": len(ds),
    }


def eval_utility_set(model, tokenizer, name, max_length, limit, mcq, eval_batch_size):
    """Compute Probability, ROUGE, Truth-Ratio-utility for one utility dataset.
    mcq=True for real_authors/world_facts (multiple-choice probability)."""
    ds = load_tofu_config(name, limit)
    probs, rouges, tr_util = [], [], []
    for i in tqdm(range(0, len(ds), eval_batch_size), desc=name):
        batch = ds[i:i + eval_batch_size]
        if isinstance(batch, dict):
            keys = list(batch.keys())
            batch = [dict(zip(keys, vals)) for vals in zip(*batch.values())]
        qs = [ex["question"] for ex in batch]
        gts = [ex["answer"] for ex in batch]
        preds = generate_answers(model, tokenizer, qs)
        for ex, q, gt, pred in zip(batch, qs, gts, preds):
            # probability
            if mcq:
                probs.append(mcq_probability(model, tokenizer, ex, max_length))
            else:
                probs.append(norm_prob(answer_nll(model, tokenizer, q, gt, max_length)))
            # rouge
            rouges.append(rougeL_recall(pred, gt))
            # truth ratio utility transform: max(0, 1 - 1/R)
            R = truth_ratio_for_example(model, tokenizer, ex, max_length)
            if R is not None and R > 0:
                tr_util.append(max(0.0, 1.0 - 1.0 / R))
    return {
        "probability": float(np.mean(probs)) if probs else 0.0,
        "rougeL_recall": float(np.mean(rouges)) if rouges else 0.0,
        "truth_ratio": float(np.mean(tr_util)) if tr_util else 0.0,
        "n": len(ds),
    }


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="checkpoint dir to evaluate (e.g. checkpoints/unlearn_npo)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--forget_split", default="forget01_perturbed")
    ap.add_argument("--out", required=True, help="output eval json path")
    ap.add_argument("--forget_quality_ref", default=None,
                    help="path to the RETAIN reference eval json (for Forget Quality KS test)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--limit_forget", type=int, default=None)
    ap.add_argument("--limit_utility", type=int, default=100)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        args.limit_forget = 4
        args.limit_utility = 4

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        bf16=False,
        fp16=True,
        weight_dtype="fp16",
    )
    model.to(DEVICE)
    model.eval()
    torch.backends.cudnn.benchmark = True

    logger.info("evaluating forget split: %s", args.forget_split)
    forget = eval_forget(model, tokenizer, args.forget_split, args.max_length, args.limit_forget, args.eval_batch_size)

    logger.info("evaluating utility: retain_perturbed")
    retain = eval_utility_set(model, tokenizer, "retain_perturbed",
                              args.max_length, args.limit_utility, mcq=False, eval_batch_size=args.eval_batch_size)
    logger.info("evaluating utility: real_authors_perturbed")
    real = eval_utility_set(model, tokenizer, "real_authors_perturbed",
                            args.max_length, args.limit_utility, mcq=True, eval_batch_size=args.eval_batch_size)
    logger.info("evaluating utility: world_facts_perturbed")
    world = eval_utility_set(model, tokenizer, "world_facts_perturbed",
                             args.max_length, args.limit_utility, mcq=True, eval_batch_size=args.eval_batch_size)

    # Model Utility = harmonic mean of 9 numbers (3 metrics x 3 sets).
    cands = [
        retain["probability"], retain["rougeL_recall"], retain["truth_ratio"],
        real["probability"], real["rougeL_recall"], real["truth_ratio"],
        world["probability"], world["rougeL_recall"], world["truth_ratio"],
    ]
    cands = [max(c, 1e-8) for c in cands]  # hmean needs strictly positive
    model_utility = float(hmean(cands))

    # Forget Quality (optional): KS test vs retain reference forget truth-ratio array.
    forget_quality = None
    if args.forget_quality_ref is not None and os.path.exists(args.forget_quality_ref):
        with open(args.forget_quality_ref) as f:
            ref = json.load(f)
        ref_tr = ref["forget"]["truth_ratio"]
        if len(ref_tr) > 0 and len(forget["truth_ratio"]) > 0:
            ks = ks_2samp(forget["truth_ratio"], ref_tr)
            forget_quality = float(ks.pvalue)
            logger.info("Forget Quality (KS p-value) = %.6f", forget_quality)

    result = {
        "model_path": args.model_path,
        "forget_split": args.forget_split,
        "forget": forget,
        "utility": {
            "retain": retain,
            "real_authors": real,
            "world_facts": world,
            "Model Utility": model_utility,
        },
        "Forget Quality": forget_quality,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Model Utility = %.4f", model_utility)
    logger.info("saved eval to %s", args.out)


if __name__ == "__main__":
    main()
