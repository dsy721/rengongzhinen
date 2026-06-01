# TODO: 模型模块，后续实现模型加载与封装。

from __future__ import annotations

import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def load_model_and_tokenizer(
    model_name: str,
    *,
    bf16: bool = False,
    fp16: bool = False,
    weight_dtype: str = "fp32",
    **kwargs,
):
    supported = torch.cuda.is_bf16_supported()
    print(supported)

    if bf16 and not supported:
        warnings.warn("bf16=true but current GPU does not support bf16; falling back to fp16.", RuntimeWarning)
        bf16 = False
        fp16 = True

    if weight_dtype == "fp16":
        torch_dtype = torch.float16
    elif weight_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, **kwargs)
    return model, tokenizer


def wrap_lora(model, r=8, alpha=16, dropout=0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, cfg)
    return model


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total
