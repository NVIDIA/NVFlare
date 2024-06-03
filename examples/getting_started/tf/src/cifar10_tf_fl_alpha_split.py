# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, callbacks, losses, ops
from tf_net import ModerateTFNet

# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

PATH = "./tf_model.weights.h5"


class SparseCategoricalCrossentropyWithFedProx(losses.SparseCategoricalCrossentropy):
    def __init__(
            self,
            *args,
            mu: float = 1e-3,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mu < 0.0:
            raise ValueError("mu should be no less than 0.0")
        self.mu = mu

        self.current_model = None
        self.target_model = None

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = super().call(y_true, y_pred)
        if self.current_model and self.target_model:
            fedprox_loss = 0
            for key in self.target_model:
                tar_layer = self.target_model[key]
                cur_layer = self.current_model[key]
                for t_w, c_w in zip(tar_layer, cur_layer):
                    fedprox_loss += ops.sum((t_w - c_w) ** 2)
            return loss + (self.mu/2) * fedprox_loss
        else:
            return loss


class SetFedProxValues(callbacks.Callback):
    def __init__(self, fedprox_loss):
        super().__init__()
        self.fedprox_loss = fedprox_loss

    def on_train_begin(self, logs=None):
        self.fedprox_loss.target_model = {layer.name: layer.get_weights() for layer in self.model.layers}

    def on_train_batch_begin(self, batch, logs=None):
        self.fedprox_loss.current_model = {layer.name: layer.get_weights() for layer in self.model.layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True
    )
    parser.add_argument(
        "--train_idx_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--fedprox_mu",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    # (2) initializes NVFlare client API
    flare.init()

    # load train indices
    print(f"Loading train indices from {args.train_idx_path}")
    train_idx = np.load(args.train_idx_path)

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    train_images = train_images[train_idx]
    train_labels = train_labels[train_idx]

    unq, unq_cnt = np.unique(train_labels, return_counts=True)
    print(f"Loaded {len(train_idx)} training indices with label distribution:")
    print("Unique labels:", unq)
    print("Unique Counts:", unq_cnt)

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = ModerateTFNet()
    model.build(input_shape=(None, 32, 32, 3))

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs", write_graph=False)]

    if args.fedprox_mu > 0:
        loss = SparseCategoricalCrossentropyWithFedProx(from_logits=True)
        callbacks.append(SetFedProxValues(loss))
    else:
        loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=loss, metrics=["accuracy"]
    )
    model.summary()

    # (2) initializes NVFlare client API
    flare.init()

    summary_writer = SummaryWriter()
    tf_summary_writer = tf.summary.create_file_writer(logdir="./logs/validation")
    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (optional) print system info
        system_info = flare.system_info()
        print(f"NVFlare system info: {system_info}")

        # (4) loads model from NVFlare
        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)

        # (5) evaluate aggregated/received model
        _, test_global_acc = model.evaluate(test_images, test_labels, verbose=2)
        summary_writer.add_scalar(tag="global_model_accuracy", scalar=test_global_acc, global_step=input_model.current_round)
        with tf_summary_writer.as_default():
            tf.summary.scalar("global_model_accuracy", test_global_acc, input_model.current_round)
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the {len(test_images)} test images: {test_global_acc * 100} %"
        )

        start_epoch = args.epochs*input_model.current_round
        end_epoch = start_epoch + args.epochs
        print(f"Train from epoch {start_epoch} to {end_epoch}")
        model.fit(train_images, train_labels, epochs=end_epoch, validation_data=(test_images, test_labels), batch_size=args.batch_size, callbacks=callbacks,
                  initial_epoch=start_epoch, validation_freq=args.epochs)

        print("Finished Training")

        model.save_weights(PATH)

        _, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        summary_writer.add_scalar(tag="local_model_accuracy", scalar=test_acc, global_step=input_model.current_round)
        with tf_summary_writer.as_default():
            tf.summary.scalar("local_model_accuracy", test_acc, input_model.current_round)
        print(f"Accuracy of the model on the {len(test_images)} test images: {test_acc * 100} %")

        # (6) construct trained FL model (A dict of {layer name: layer weights} from the keras model)
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers}, metrics={"accuracy": test_global_acc}
        )
        # (7) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
