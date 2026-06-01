# TODO: 记忆验证脚本，仅用于对比 target 与 retain_ref 在 forget01 上的生成结果。

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from src.data import load_tofu
from src.model import load_model_and_tokenizer


def generate_answer(model, tokenizer, question) -> str:
    messages = [{"role": "user", "content": question}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    gen_ids = out[0][enc["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main() -> None:
    device = "cuda"
    ds = load_tofu("forget01")
    samples = ds.select(range(5))

    target_answers = []
    retain_answers = []

    model_t, tok = load_model_and_tokenizer(
        "checkpoints/target",
        bf16=False,
        fp16=False,
        weight_dtype="fp16",
    )
    model_t.eval().to(device)
    for sample in samples:
        target_answers.append(generate_answer(model_t, tok, sample["question"]))
    del model_t
    torch.cuda.empty_cache()

    model_r, tok = load_model_and_tokenizer(
        "checkpoints/retain_ref",
        bf16=False,
        fp16=False,
        weight_dtype="fp16",
    )
    model_r.eval().to(device)
    for sample in samples:
        retain_answers.append(generate_answer(model_r, tok, sample["question"]))
    del model_r
    torch.cuda.empty_cache()

    for i in range(5):
        print("=" * 60)
        print(f"====== 样本 {i + 1} ======")
        print(f"问题:         {samples[i]['question']}")
        print(f"真值答案:     {samples[i]['answer']}")
        print(f"target 回答:     {target_answers[i]}")
        print(f"retain_ref 回答: {retain_answers[i]}")


if __name__ == "__main__":
    main()
