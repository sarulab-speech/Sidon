"""Execution options that keep parameter names and training state: fused AdamW, block compilation, resume."""

from copy import deepcopy

import torch
from torch import nn


def adamw(parameters, *, fused: bool | None = None, **kwargs) -> torch.optim.AdamW:
    """AdamW whose requested implementation (fused) and betas also hold after loading a saved optimizer state,
    which would otherwise restore the saved ones."""
    optimizer = torch.optim.AdamW(parameters, fused=fused, **kwargs)
    overrides = {}
    if fused is not None:
        overrides.update(fused=fused, foreach=None)
    if "betas" in kwargs:
        overrides["betas"] = tuple(kwargs["betas"])
    if overrides:
        def restore_requested_options(optimizer, state_dict):
            # Applied before loading, so fused step counters are placed on the parameter device while moments,
            # step counts and learning rates are kept.
            return {**state_dict, "param_groups": [{**group, **overrides} for group in state_dict["param_groups"]]}

        optimizer.register_load_state_dict_pre_hook(restore_requested_options)
    return optimizer


def compile_training_blocks(model: nn.Module, *, backend: str = "inductor") -> int:
    """Apply model.cfg["performance"] in place: compile_ssl compiles every w2v-BERT encoder layer (dynamic lengths),
    flex_ssl_attention lets attention without gradients (the frozen VAE encoder) use FlexAttention when compiled.
    Returns the number of compiled modules."""
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertEncoderLayer

    from .ssl_attention import RelativeKeyAttention

    options = model.cfg.get("performance", {}) or {}
    for module in model.modules():
        if isinstance(module, RelativeKeyAttention):
            module.flex = bool(options.get("flex_ssl_attention", False))
    if options.get("compile_ssl", False):
        # Layers are compiled one by one, so DDP still overlaps gradient communication between them; its graph
        # splitting is not needed (and fails with symbolic batch sizes in torch 2.8).
        torch._dynamo.config.optimize_ddp = False
    count = 0
    for module in model.modules():
        if (options.get("compile_ssl", False) and isinstance(module, Wav2Vec2BertEncoderLayer)
                and module._compiled_call_impl is None):
            # Module.compile keeps parameter identities and state-dict keys.
            module.compile(backend=backend, dynamic=True)
            count += 1
    return count


# Optimizer settings a resumed separation run takes from the new config (the rest comes from the checkpoint).
RESUMED_OPTIM_KEYS = ("skip_grad_norm", "betas", "cooldown_start", "min_lr_ratio", "max_steps")


def apply_resume_options(model: nn.Module, requested: dict) -> None:
    """Options a resumed run takes from the requested config instead of the checkpoint: the performance section,
    optim.fused and, for the separation model, the learning-rate tail (cooldown_start, min_lr_ratio, max_steps),
    skip_grad_norm and betas; for the VAE stage, decoder_crop_frames. Weights and every architecture setting come
    from the checkpoint."""
    cfg = model.cfg
    cfg["performance"] = deepcopy(requested.get("performance", {}))
    if "fused" in requested.get("optim", {}):
        cfg["optim"]["fused"] = requested["optim"]["fused"]
    if hasattr(model, "readout"):
        for key in RESUMED_OPTIM_KEYS:
            if key in requested.get("optim", {}):
                cfg["optim"][key] = deepcopy(requested["optim"][key])
        enabled = bool((requested.get("conditioner") or {}).get("gradient_checkpointing", False))
        cfg["conditioner"]["gradient_checkpointing"] = enabled
        if enabled:
            model.conditioner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.conditioner.gradient_checkpointing_disable()
    elif "decoder_crop_frames" in requested:
        cfg["decoder_crop_frames"] = requested["decoder_crop_frames"]
    model.save_hyperparameters({"cfg": cfg}, logger=False)
