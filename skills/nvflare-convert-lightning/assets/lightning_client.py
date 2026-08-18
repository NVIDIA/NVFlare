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
FLARE server. Except for Cyclic, the patched callback captures metrics from this
explicit pre-fit validation and attaches them to the outgoing training result.
Do not duplicate them under ``model.__fl_meta__[MetaKey.INITIAL_METRICS]``.
"""

import math

import nvflare.client.lightning as flare


def _scalar_validation_metrics(validation_results):
    if not validation_results:
        raise RuntimeError("Lightning validation returned no metrics")

    metrics = {}
    for result in validation_results:
        if not isinstance(result, dict):
            raise RuntimeError("Lightning validation results must be dictionaries of scalar metrics")
        for key, value in result.items():
            if key in metrics:
                raise RuntimeError(f"Lightning validation returned duplicate metric key {key!r}")
            item_fn = getattr(value, "item", None)
            if callable(item_fn):
                value = item_fn()
            try:
                scalar = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"Lightning validation metric {key!r} is not scalar") from exc
            if not math.isfinite(scalar):
                raise RuntimeError(f"Lightning validation metric {key!r} is not finite")
            metrics[str(key)] = scalar

    if not metrics:
        raise RuntimeError("Lightning validation returned no scalar metrics")
    return metrics


def validate_global_model(trainer, model, datamodule=None, dataloaders=None):
    """Validate the received global model and return the trainer callback metrics.

    Call this before ``trainer.fit`` inside the round loop. Metrics come from
    the ``LightningModule``'s ``self.log(...)`` calls. The patched callback
    sends them with the training result even when the selected executor leaves
    ``train_with_evaluation`` disabled. Log any higher-is-better companion such
    as ``neg_val_loss`` inside the module's validation lifecycle so the callback
    captures it under the exact key selected by the recipe.
    """
    if datamodule is not None:
        validation_results = trainer.validate(model, datamodule=datamodule)
    else:
        validation_results = trainer.validate(model, dataloaders=dataloaders)

    return _scalar_validation_metrics(validation_results)


def should_evaluate_before_train(recipe_algorithm):
    """Return whether the selected recipe evaluates the received model before training.

    Cyclic intentionally persists its final sequential model and has no
    best-model selection. Every other supported Lightning recipe evaluates the
    received model for its server metric contract.
    """
    return recipe_algorithm != "cyclic"


def main(model, datamodule, trainer_factory, recipe_algorithm="fedavg", evaluate_only=False):
    """Lightning Client API round loop with validate-before-fit.

    ``trainer_factory`` constructs the source project's ``Trainer``. Pass the
    normalized ``algorithm`` value returned by ``nvflare recipe show`` as
    ``recipe_algorithm``; only ``cyclic`` training skips pre-fit validation. Set
    ``evaluate_only=True`` for FedEval / evaluation-only conversions: regardless
    of the selected algorithm, the round runs ``trainer.validate`` so the patched
    trainer sends validation metrics, and skips local training. Do not call
    ``trainer.fit`` in that mode.

    When best-model selection needs a higher-is-better companion such as
    ``neg_val_loss``, log it from the module's validation lifecycle under the
    exact key selected by the recipe.
    """
    trainer = trainer_factory()
    flare.patch(trainer)
    evaluate_before_train = should_evaluate_before_train(recipe_algorithm)

    while flare.is_running():
        # receive() is optional metadata/task-progression access only; the
        # patched trainer loads the global model internally.
        flare.receive()
        if evaluate_only or evaluate_before_train:
            validate_global_model(trainer, model, datamodule=datamodule)
        if evaluate_only:
            continue
        trainer.fit(model, datamodule=datamodule)
