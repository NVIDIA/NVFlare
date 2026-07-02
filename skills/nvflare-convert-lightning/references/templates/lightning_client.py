# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def main(model, datamodule, trainer_factory):
    """Lightning Client API round loop with validate-before-fit.

    ``trainer_factory`` constructs the source project's ``Trainer``.
    """
    trainer = trainer_factory()
    flare.patch(trainer)

    while flare.is_running():
        # receive() is optional metadata/task-progression access only; the
        # patched trainer loads the global model internally.
        flare.receive()
        validate_global_model(trainer, model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
