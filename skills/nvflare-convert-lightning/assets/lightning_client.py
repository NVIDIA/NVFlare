# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged Lightning Client API conversion template: patch + native evaluation.

Copy and adapt this into a generated ``client.py``. The patched trainer owns
model exchange; do not generate a manual ``FLModel`` send/receive path and do
not pass the received ``input_model`` into ``Trainer`` methods. Evaluation
stays inside Lightning (``validation_step`` / ``self.log`` /
``trainer.validate``); do not generate a raw PyTorch ``model.eval()`` loop.

``validate_global_model`` is factored out so a generated conversion can be
validated against a toy ``LightningModule`` and dataloader without a running
FLARE server.
"""

import nvflare.client.lightning as flare


def validate_global_model(trainer, model, datamodule=None, dataloaders=None):
    """Validate the received global model and return the trainer callback metrics.

    Call this before ``trainer.fit`` inside the round loop so each round reports
    global-model metrics that the server can use for model selection. Metrics
    come from the ``LightningModule``'s ``self.log(...)`` calls.
    """
    if datamodule is not None:
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.validate(model, dataloaders=dataloaders)
    return dict(trainer.callback_metrics)


def main(model, datamodule, trainer_factory, evaluate_only=False):
    """Lightning Client API round loop with validate-before-fit.

    ``trainer_factory`` constructs the source project's ``Trainer``. Set
    ``evaluate_only=True`` for FedEval / evaluation-only conversions: the round
    runs ``trainer.validate`` so the patched trainer sends validation metrics,
    and skips local training. Do not call ``trainer.fit`` in that mode.
    """
    trainer = trainer_factory()
    flare.patch(trainer)

    while flare.is_running():
        # receive() is optional metadata/task-progression access only; the
        # patched trainer loads the global model internally.
        flare.receive()
        validate_global_model(trainer, model, datamodule=datamodule)
        if evaluate_only:
            continue
        trainer.fit(model, datamodule=datamodule)
