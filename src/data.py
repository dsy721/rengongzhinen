# TODO: 数据模块，负责加载 TOFU 数据集并把 question/answer 转成只在 answer 上计 loss 的训练样本。

from __future__ import annotations

from typing import Dict

from datasets import Dataset, load_dataset


def load_tofu(split: str):
    # 这里的 split 实际是 TOFU 的配置名，真正的 split 固定是 "train"
    return load_dataset("locuslab/TOFU", name=split, split="train")


def to_chat_text(question, answer, tokenizer) -> dict:
    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = list(prompt_ids) + list(answer_ids)
    labels = input_ids.copy()
    for i in range(len(prompt_ids)):
        labels[i] = -100

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_dataset(split, tokenizer, max_length) -> Dataset:
    dataset = load_tofu(split)

    def _map_fn(example):
        item = to_chat_text(example["question"], example["answer"], tokenizer)
        input_ids = item["input_ids"][:max_length]
        labels = item["labels"][:max_length]
        attention_mask = item["attention_mask"][:max_length]

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return dataset.map(_map_fn, remove_columns=dataset.column_names)
