# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged Hugging Face Client API conversion template: patch + Trainer loop.

Copy and adapt this into a generated ``client.py``. ``trainer_factory`` builds
the source project's model, tokenizer/processor, datasets, collator, arguments,
callbacks, and Trainer once after FLARE initialization. The patched Trainer
owns model exchange; do not add a manual ``FLModel`` receive/send path.

Keep ``evaluate_before_train=True`` when source-backed per-round evaluation or
best-model selection is required. Set it to ``False`` only for a valid
train-only source path.

The entry is rankless for CPU, single-GPU, and other single-process training.
With an initialized multi-process group, it passes
``torch.distributed.get_rank()`` to FLARE. If environment markers declare a
multi-process launch before the process group is initialized, global ``RANK``
must already be available; otherwise the entry fails before FLARE initialization.
For a preserved multi-GPU launch, follow
``references/huggingface-state-and-distributed.md``.
"""

import os

import torch.distributed as dist

import nvflare.client.hf as flare


_MULTIRANK_SIZE_ENV_VARS = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS")


def _environment_declares_multirank() -> bool:
    for name in _MULTIRANK_SIZE_ENV_VARS:
        try:
            if int(os.environ.get(name, "1") or 1) > 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _has_valid_global_rank_env() -> bool:
    try:
        int(os.environ["RANK"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def make_hf_argument_parser(dataclass_types):
    """Create a strict parser for generated clients that use Hugging Face dataclasses."""
    from transformers import HfArgumentParser

    return HfArgumentParser(dataclass_types, allow_abbrev=False)


def main(trainer_factory, *, evaluate_before_train=True):
    """Run one persistent patched Trainer in single-process or distributed mode."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        flare.init(rank=dist.get_rank())
    else:
        if _environment_declares_multirank() and not _has_valid_global_rank_env():
            raise RuntimeError(
                "multi-process launch detected but global RANK is unavailable; initialize torch.distributed "
                "or export a valid global RANK before FLARE Client API initialization"
            )
        # CPU and single-GPU runs stay rankless. Under torchrun, flare.init()
        # resolves the global RANK environment value.
        flare.init()
    trainer = trainer_factory()
    flare.patch(trainer)

    while flare.is_running():
        if evaluate_before_train:
            trainer.evaluate()
        trainer.train()
