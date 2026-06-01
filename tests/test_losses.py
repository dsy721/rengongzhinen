# TODO: 损失函数测试占位文件，后续补充单元测试。

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.losses import grad_diff_forget_loss, npo_forget_loss, retain_ce_loss, seq_logp_sum


def _run_and_report(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        raise


def make_tiny_model():
    config = GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_ctx=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def make_batch():
    input_ids = torch.tensor(
        [[1, 5, 6, 7, 2, 0], [1, 8, 9, 10, 11, 2]],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[:, :3] = -100
    attention_mask = (input_ids != 0).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def test_retain_and_graddiff_losses_backward():
    model = make_tiny_model()
    batch = make_batch()

    loss_retain = retain_ce_loss(model, batch)
    assert loss_retain.item() > 0
    loss_retain.backward()
    assert any(p.grad is not None for p in model.parameters())

    model.zero_grad()
    loss_graddiff = grad_diff_forget_loss(model, batch)
    assert loss_graddiff.item() < 0
    loss_graddiff.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_npo_loss_positive_finite_and_backward():
    model = make_tiny_model()
    ref_model = make_tiny_model()
    ref_model.load_state_dict(model.state_dict())
    batch = make_batch()

    loss = npo_forget_loss(model, ref_model, batch, beta=0.1)
    assert loss.item() > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_npo_self_reference_matches_closed_form():
    model = make_tiny_model()
    batch = make_batch()
    beta = 0.1
    loss = npo_forget_loss(model, model, batch, beta=beta)
    expected = (2.0 / beta) * math.log(2.0)
    assert abs(loss.item() - expected) < 0.5


if __name__ == "__main__":
    _run_and_report("test_retain_and_graddiff_losses_backward", test_retain_and_graddiff_losses_backward)
    _run_and_report("test_npo_loss_positive_finite_and_backward", test_npo_loss_positive_finite_and_backward)
    _run_and_report("test_npo_self_reference_matches_closed_form", test_npo_self_reference_matches_closed_form)
