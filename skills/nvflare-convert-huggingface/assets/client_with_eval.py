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
"""

import nvflare.client.hf as flare


def main(trainer_factory, rank=0, evaluate_before_train=True):
    """Run one persistent patched Trainer across federated rounds."""
    flare.init(rank=rank)
    trainer = trainer_factory()
    flare.patch(trainer)

    while flare.is_running():
        if evaluate_before_train:
            trainer.evaluate()
        trainer.train()
