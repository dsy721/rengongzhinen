# TODO: 损失函数模块，提供 TOFU 阶段所需的 retain / GradDiff / NPO 损失。

from __future__ import annotations

import torch
import torch.nn.functional as F


def retain_ce_loss(model, batch) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss


def grad_diff_forget_loss(model, batch) -> torch.Tensor:
    return -retain_ce_loss(model, batch)


def seq_logp_sum(m, batch) -> torch.Tensor:
    logits = m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]

    vocab_size = shift_logits.size(-1)
    nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    nll = nll.view(shift_labels.shape)

    mask = shift_labels != -100
    sum_nll = (nll * mask).sum(dim=1)
    return -sum_nll


def npo_forget_loss(model, ref_model, batch, beta: float = 0.1) -> torch.Tensor:
    logp_theta = seq_logp_sum(model, batch)
    with torch.no_grad():
        logp_ref = seq_logp_sum(ref_model, batch)
    loss = (2.0 / beta) * F.softplus(beta * (logp_theta - logp_ref)).mean()
    return loss
