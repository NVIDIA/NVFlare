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

"""Test-only observers; the FedOpt algorithm and hello-pt trainer are unchanged."""

import os
from pathlib import Path

import numpy as np
import torch

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType


def snapshot(value):
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return torch.as_tensor(value).detach().cpu().clone()
    return value


class FedOptObserver(FLComponent):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.diff = None

    def handle_event(self, event_type, fl_ctx):
        if event_type == AppEventType.AFTER_AGGREGATION:
            dxo = from_shareable(fl_ctx.get_prop(AppConstants.AGGREGATION_RESULT))
            assert dxo.data_kind == DataKind.WEIGHT_DIFF
            self.diff = snapshot(dxo.data)
        elif event_type == AppEventType.AFTER_SHAREABLE_TO_LEARNABLE:
            generator = fl_ctx.get_engine().get_component("shareable_generator")
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            folder = Path(self.output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(
                snapshot(
                    {
                        "diff": self.diff,
                        "model": generator.model.state_dict(),
                        "optimizer": generator.optimizer.state_dict(),
                        "scheduler": generator.lr_scheduler.state_dict(),
                        "pid": os.getpid(),
                    }
                ),
                folder / f"round_{current_round}.pt",
            )


def observe_client():
    import client as trainer

    import nvflare.client as flare

    original_receive, original_send = flare.receive, flare.send
    original_train = trainer.train_one_round
    training = {}
    folder = Path("fedopt_evidence")
    folder.mkdir(exist_ok=True)

    def receive(*args, **kwargs):
        model = original_receive(*args, **kwargs)
        if flare.is_train():
            training["round"] = model.current_round
            torch.save(snapshot(model.params), folder / f"{model.current_round}_input.pt")
        return model

    def train(model, optimizer, train_loader, *args, **kwargs):
        training.update(optimizer=optimizer, train_loader=train_loader)
        return original_train(model, optimizer, train_loader, *args, **kwargs)

    def send(model, *args, **kwargs):
        if flare.is_train():
            torch.save(
                snapshot(
                    {
                        "params": model.params,
                        "metrics": model.metrics,
                        "steps": model.meta["NUM_STEPS_CURRENT_ROUND"],
                        "optimizer": training["optimizer"].state_dict(),
                        "train_rng": training["train_loader"].generator.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "pid": os.getpid(),
                    }
                ),
                folder / f"{training['round']}_output.pt",
            )
        return original_send(model, *args, **kwargs)

    flare.receive, flare.send = receive, send
    trainer.train_one_round = train
    trainer.main()


if __name__ == "__main__":
    observe_client()
