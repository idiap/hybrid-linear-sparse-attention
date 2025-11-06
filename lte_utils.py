#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from einops import rearrange
from torch.distributed._functional_collectives import all_reduce, c10d
from torch.nn.functional import relu

import math

def initialize_lte(
    model: torch.nn.Module,
    train_state: Any,
    job_config: Any,
    logger: Any,
) -> bool:
    if not (hasattr(model, "model") and getattr(model.model, "use_lte", False)):
        # Nothing to do – LTE is switched off in this model.
        return False

    router_modules = [
        module for name, module in model.named_modules()
        if name.endswith(".router")
    ]
    if train_state.step <= model.config.attn["l1_warmup_steps"]:
        for module in router_modules:
            module.enabled = False
            module.routing_grad = False
        model.model.num_kept.fill_(job_config.training.seq_len)
        logger.info("Routers are disabled at step %d (until warmup steps %d)" % (
            train_state.step,
            model.config.attn["l1_warmup_steps"],
        ))
    else:
        for module in router_modules:
            module.enabled = True
            module.routing_grad = model.config.attn["routing_grad"]
        logger.info("Routers are enabled at step %d" % train_state.step)

    if train_state.step > model.config.attn["routing_freeze_steps"]:
        for module in router_modules:
            for p in module.parameters():
                p.requires_grad = True
        logger.info("Router grads are enabled at step %d (routing_freeze_steps = %d)" % (
            train_state.step,
            model.config.attn["routing_freeze_steps"],
        ))
    else:
        for module in router_modules:
            for p in module.parameters():
                p.requires_grad = False
        logger.info("Router grads are disabled at step %d (routing_freeze_steps = %d)" % (
            train_state.step,
            model.config.attn["routing_freeze_steps"],
        ))


    if train_state.step <= model.config.attn["l1_lambda_initial_steps"]:
        model.model.l1_lambda.fill_(0.0)

    n_lte_params = sum(
        p.numel() for n, p in model.named_parameters() if "router" in n
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Number of LTE parameters: {n_lte_params:,} ({n_lte_params / total_params:.2%})"
    )

    return True

def prepare_lte_step(model: torch.nn.Module, train_state: Any, logger: Any) -> Dict[str, Any]:
    if train_state.step == model.config.attn["l1_warmup_steps"] + 1:
        logger.info("Enabled routers at step %d", train_state.step)
        for name, module in model.named_modules():
            if name.endswith(".router"):
                module.enabled = True
                module.routing_grad = model.config.attn["routing_grad"]


    if train_state.step == model.config.attn["routing_freeze_steps"] + 1:
        for name, module in model.named_modules():
            if name.endswith(".router"):
                for p in module.parameters():
                    p.requires_grad = True
        logger.info("Enabled router gradients at step %d", train_state.step)

    l1_lambda = model.model.l1_lambda

    # For layer‑shared λ we need an extra "head" dim so the broadcast works.
    if model.config.attn["l1_lambda_shared"] == "layer":
        l1_lambda = l1_lambda.unsqueeze(-1)

    return {
        "l1_lambda": l1_lambda,
        "all_l1_losses": [],  # type: List[torch.Tensor]
        "all_l1_scores": [],  # type: List[torch.Tensor]
        "layer_sparsity": defaultdict(list),  # type: Dict[int,List[torch.Tensor]]
    }

def apply_lte_loss(
    output: Any,
    lte_state: Dict[str, Any],
) -> int:
    l1_lambda = lte_state["l1_lambda"]
    sparsity, masks, scores = output[-1]  # Nl × {B,L,H} each

    for layer_idx, sp in sparsity.items():
        lte_state["layer_sparsity"][layer_idx].append(sp.detach().mean(0))

    all_scores: List[torch.Tensor] = [relu(s) for s in scores.values()] # Only keep positive scores for L1 loss
    all_scores = torch.stack(all_scores)  # [Nl,B,L,H]
    all_scores = rearrange(all_scores, "n b l h -> b l n h")  # [B,L,Nl,H]

    length = all_scores.size(1)
    l1_score = all_scores.mean()
    l1_loss = (l1_lambda * all_scores).mean()

    lte_state["all_l1_scores"].append(l1_score.detach())
    lte_state["all_l1_losses"].append(l1_loss.detach())

    output.loss = output.loss + l1_loss

    return length

def multiplicative_update(model, kept, train_state, update_cycle):
    l, r = model.config.attn["l1_lambda_tolerate"]
    is_large = kept > model.config.attn["buffer_size"] * l
    is_small = kept < model.config.attn["buffer_size"] * r

    sign = is_large.to(torch.int) - is_small.to(torch.int)

    if train_state.step // update_cycle == model.config.attn["l1_lambda_initial_steps"] // update_cycle + 1:
        model.model.l1_lambda.copy_(torch.where(is_large, model.config.attn["l1_lambda_init"], 0.0))
    else:
        coeff = torch.where(sign > 0, model.config.attn["l1_alpha"], 1 / model.config.attn["l1_alpha_neg"])
        coeff = torch.where(sign == 0, 1., coeff)  # [L] or [L, H]
        model.model.l1_lambda.mul_(coeff).clamp_(max=model.config.attn["l1_lambda_max"])
        model.model.l1_lambda.copy_(torch.where(model.model.l1_lambda < model.config.attn["l1_lambda_min"],
                                                0.0,
                                                model.model.l1_lambda))
        model.model.l1_lambda.copy_(torch.where(
            (model.model.l1_lambda == 0) & is_large,
            model.config.attn["l1_lambda_min"],
            model.model.l1_lambda)
        )

def update_lte_lambda(
    model: torch.nn.Module,
    train_state: Any,
    lte_state: Dict[str, Any],
    length: int,
    parallel_dims: Any,
    world_mesh: Any,
    logger: Any,
) -> Dict[str, float]:
    l1_loss = sum(lte_state["all_l1_losses"]) / len(lte_state["all_l1_losses"])
    l1_score = sum(lte_state["all_l1_scores"]) / len(lte_state["all_l1_scores"])

    global_l1_loss = l1_loss.item()
    global_l1_score = l1_score.item()

    for k in lte_state["layer_sparsity"]:
        lte_state["layer_sparsity"][k] = torch.stack(lte_state["layer_sparsity"][k]).mean(0)

    overall_sparsity = torch.stack(list(lte_state["layer_sparsity"].values())).mean()

    all_kept = torch.stack(list(lte_state["layer_sparsity"].values())) * length
    update_cycle = model.config.attn["l1_lambda_update_steps"]
    ema_alpha = 2 / (1 + update_cycle // 2)
    model.model.num_kept.mul_(1 - ema_alpha).add_(all_kept * ema_alpha)

    if train_state.step <= model.config.attn["l1_lambda_initial_steps"] or train_state.step % update_cycle != 0:
        return {
            "overall_sparsity": overall_sparsity,
            "global_l1_loss": global_l1_loss,
            "global_l1_score": global_l1_score,
        }

    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        updated_kept = all_reduce(
            model.model.num_kept,
            reduceOp=c10d.ReduceOp.AVG.name,
            group=world_mesh["dp_cp"],
        )
        model.model.num_kept.copy_(updated_kept)

    match model.config.attn["l1_lambda_shared"]:
        case "all":
            kept = model.model.num_kept.mean()
        case "layer":
            kept = model.model.num_kept.mean(-1)  # [L]
        case "head":
            kept = model.model.num_kept  # [L,H]

    multiplicative_update(model, kept, train_state, update_cycle)

    return {
        "overall_sparsity": overall_sparsity,
        "global_l1_loss": global_l1_loss,
        "global_l1_score": global_l1_score,
    }

def make_lte_metrics(
    train_state_step: int,
    lte_state: Dict[str, Any],
    log_payload: Dict[str, float],
    model: torch.nn.Module,
) -> Dict[str, float | torch.Tensor]:
    metrics: Dict[str, float | torch.Tensor] = {
        "lte/overall_sparsity": log_payload["overall_sparsity"],
        "lte/l1_loss": log_payload["global_l1_loss"],
        "lte/l1_score": log_payload["global_l1_score"],
    }

    for i, (k, sparse) in enumerate(lte_state["layer_sparsity"].items()):
        metrics[f"lte_layer/sparsity_{k}"] = sparse.mean().item()
        metrics[f"lte_layer/num_kept_{k}"] = model.model.num_kept[i].mean().item()

    if train_state_step % model.config.attn["l1_lambda_update_steps"] == 0:
        metrics["lte/overall_l1_lambda"] = model.model.l1_lambda.mean().item()

        for i, (k, sparse) in enumerate(lte_state["layer_sparsity"].items()):
            if model.config.attn["l1_lambda_shared"] != "all":
                metrics[f"lte_layer/l1_lambda_{k}"] = model.model.l1_lambda[i].mean().item()

            metrics[f"lte_head/sparsity_{k}"] = sparse
            metrics[f"lte_head/num_kept_{k}"] = model.model.num_kept[i]
            metrics[f"lte_head/num_kept_max_{k}"] = model.model.num_kept[i].max().item()
            if model.config.attn["l1_lambda_shared"] == "head":
                metrics[f"lte_head/l1_lambda_{k}"] = model.model.l1_lambda[i]
                metrics[f"lte_head/l1_lambda_max_{k}"] = model.model.l1_lambda[i].max().item()

    return metrics
