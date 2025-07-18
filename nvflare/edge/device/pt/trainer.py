# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn as nn

from nvflare.apis.dxo import DXO
from nvflare.edge.device.defs import Context, ContextKey, DataSource, EventType, Executor, Signal, Transform


class PTTrainer(Executor):

    def __init__(self, epoch: int, lr, loss_fn, optimizer, transforms):
        self.epoch = epoch
        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.transforms = transforms

    def execute(self, task_data: DXO, ctx: Context, abort_signal: Signal) -> DXO:
        # the model must have been converted to nn.Module by some filters
        model = task_data.data.get("model")
        assert isinstance(model, nn.Module)
        params = task_data.data.get("params")
        data_source = ctx.get(ContextKey.DATA_SOURCE)
        assert isinstance(data_source, DataSource)

        # load the dataset
        train_dataset = data_source.get_dataset(dataset_type="train", ctx=ctx)

        # loss function and optimizer
        lr = params.get("learning_rate")
        if not lr:
            lr = self.lr

        optimizer = self.optimizer.get(model.parameters(), lr=lr)

        batch_size = params.get("batch_size")
        if not batch_size:
            batch_size = 10

        n_epochs = params.get("num_epochs")
        if not n_epochs:
            n_epochs = self.epoch

        batches_per_epoch = (train_dataset.size() + batch_size - 1) / batch_size

        for epoch in range(n_epochs):
            for i in range(batches_per_epoch):
                batch = train_dataset.get_next_batch(batch_size)

                if self.transforms:
                    for t in self.transforms:
                        assert isinstance(t, Transform)
                        batch = t.transform(batch, ctx, abort_signal)

                x_batch = batch.get_input()
                y_batch = batch.get_label()

                # forward pass
                y_pred = model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)

                ctx.fire_event(
                    EventType.LOSS_GENERATED,
                    data={
                        "loss": float(loss),
                        "epoch": epoch,
                        "iter": i,
                    },
                    abort_signal=abort_signal,
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

            # reset dataset for next epoch
            train_dataset.reset()

        weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights[name] = param.data

        return DXO(
            data_kind="model",
            data={
                "weights": weights,
            },
            meta={
                "dataset_size": train_dataset.size(),
                "batch_size": batch_size,
                "num_epochs": n_epochs,
                "lr": lr,
            },
        )
