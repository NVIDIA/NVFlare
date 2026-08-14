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
FLARE server. It also preserves the validation result on the patched
LightningModule's supported metadata channel. This is required for training
recipes whose executor does not attach callback metrics to the outgoing model.
"""

import math

import nvflare.client.lightning as flare
from nvflare.app_common.abstract.fl_model import MetaKey


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


def add_higher_is_better_metrics(metrics, make_higher_is_better):
    """Derive a higher-is-better ``neg_<key>`` companion for each named metric.

    ``key_metric`` selects on higher-is-better values only, so what the client
    delivers and the recipe selects must be a higher-is-better value. Name the
    source metrics whose direction must be flipped — typically a loss — and this
    returns them alongside a negated companion that *is* higher-is-better:
    ``("val_loss",)`` produces ``neg_val_loss``, selected as
    ``key_metric="neg_val_loss"``.

    This is the Lightning counterpart of the framework-neutral rule in
    ``../../nvflare-shared/references/pytorch-family-recipe-construction.md``.

    Pass only source-backed keys whose direction the source establishes. Returns
    a new dict; the original metric is preserved alongside the companion and the
    input mapping is left unchanged.
    """
    result = dict(metrics)
    for key in make_higher_is_better:
        if key not in result:
            raise RuntimeError(f"metric {key!r} is not in the validation results")
        companion = f"neg_{key}"
        if companion in result:
            raise RuntimeError(f"higher-is-better companion {companion!r} already exists")
        result[companion] = -result[key]
    return result


def validate_global_model(trainer, model, datamodule=None, dataloaders=None, make_higher_is_better=()):
    """Validate the received global model and return the trainer callback metrics.

    Call this before ``trainer.fit`` inside the round loop. Metrics come from
    the ``LightningModule``'s ``self.log(...)`` calls. Preserve them under
    ``MetaKey.INITIAL_METRICS`` so the patched callback sends them with the
    training result even when the selected executor leaves
    ``train_with_evaluation`` disabled.

    Pass ``make_higher_is_better`` for source metrics whose direction must be
    flipped before selection, such as ``("val_loss",)``. Each gains a ``neg_``
    companion that is higher-is-better; select that companion with
    ``key_metric``, never the original.
    """
    if datamodule is not None:
        validation_results = trainer.validate(model, datamodule=datamodule)
    else:
        validation_results = trainer.validate(model, dataloaders=dataloaders)

    metrics = _scalar_validation_metrics(validation_results)
    if make_higher_is_better:
        metrics = add_higher_is_better_metrics(metrics, make_higher_is_better)
    fl_meta = getattr(model, "__fl_meta__", {})
    if not isinstance(fl_meta, dict):
        raise RuntimeError("LightningModule.__fl_meta__ must be a dictionary")
    model.__fl_meta__ = dict(fl_meta)
    model.__fl_meta__[MetaKey.INITIAL_METRICS] = metrics
    return metrics


def main(model, datamodule, trainer_factory, evaluate_only=False, make_higher_is_better=()):
    """Lightning Client API round loop with validate-before-fit.

    ``trainer_factory`` constructs the source project's ``Trainer``. Set
    ``evaluate_only=True`` for FedEval / evaluation-only conversions: the round
    runs ``trainer.validate`` so the patched trainer sends validation metrics,
    and skips local training. Do not call ``trainer.fit`` in that mode.

    Pass ``make_higher_is_better`` when best-model selection uses a source
    metric whose direction must be flipped, for example ``("val_loss",)``. Each
    named metric gains a higher-is-better ``neg_`` companion that the recipe
    selects with ``key_metric``; keep this threaded through when adapting the
    loop, or the higher-is-better key never reaches the server.
    """
    trainer = trainer_factory()
    flare.patch(trainer)

    while flare.is_running():
        # receive() is optional metadata/task-progression access only; the
        # patched trainer loads the global model internally.
        flare.receive()
        validate_global_model(
            trainer,
            model,
            datamodule=datamodule,
            make_higher_is_better=make_higher_is_better,
        )
        if evaluate_only:
            continue
        trainer.fit(model, datamodule=datamodule)
