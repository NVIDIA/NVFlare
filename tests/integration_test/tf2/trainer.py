# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import tensorflow as tf

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from .net import Net


class SimpleTrainer(Executor):
    def __init__(self, epochs_per_round):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.train_images, self.train_labels = None, None
        self.test_images, self.test_labels = None, None
        self.model = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def setup(self, fl_ctx: FLContext):
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = tf.keras.datasets.mnist.load_data()
        self.train_images, self.test_images = (
            self.train_images / 255.0,
            self.test_images / 255.0,
        )

        # simulate separate datasets for each client by dividing MNIST dataset in half
        client_name = fl_ctx.get_identity_name()
        if client_name == "site-1":
            self.train_images = self.train_images[: len(self.train_images) // 2]
            self.train_labels = self.train_labels[: len(self.train_labels) // 2]
            self.test_images = self.test_images[: len(self.test_images) // 2]
            self.test_labels = self.test_labels[: len(self.test_labels) // 2]
        elif client_name == "site-2":
            self.train_images = self.train_images[len(self.train_images) // 2 :]
            self.train_labels = self.train_labels[len(self.train_labels) // 2 :]
            self.test_images = self.test_images[len(self.test_images) // 2 :]
            self.test_labels = self.test_labels[len(self.test_labels) // 2 :]

        model = Net()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        _ = model(tf.keras.Input(shape=(28, 28)))
        self.var_list = [model.get_layer(index=index).name for index in range(len(model.get_weights()))]
        self.model = model

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: dispatched task
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.OK)

        if task_name != "train":
            return shareable

        dxo = from_shareable(shareable)
        model_weights = dxo.data

        # use previous round's client weights to replace excluded layers from server
        prev_weights = {
            self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())
        }
        print("dxo")
        ordered_model_weights = {key: model_weights.get(key) for key in prev_weights}
        for key in self.var_list:
            value = ordered_model_weights.get(key)
            if np.all(value == 0):
                ordered_model_weights[key] = prev_weights[key]

        # update local model weights with received weights
        self.model.set_weights(list(ordered_model_weights.values()))

        # adjust LR or other training time info as needed
        # such as callback in the fit function
        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=self.epochs_per_round,
            validation_data=(self.test_images, self.test_labels),
        )

        # report updated weights in shareable
        weights = {self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")

        return dxo.update_shareable(shareable)
