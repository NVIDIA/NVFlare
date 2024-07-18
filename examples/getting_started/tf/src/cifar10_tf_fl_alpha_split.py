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
    """
    Override SparseCategoricalCrossentropy loss for FedProx,
    adding regularization term.
    """
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
    """
    Set up current model and target model for FedProx loss computation.
    """
    def __init__(self, fedprox_loss):
        super().__init__()
        self.fedprox_loss = fedprox_loss

    def on_train_begin(self, logs=None):
        self.fedprox_loss.target_model = {layer.name: layer.get_weights() for layer in self.model.layers}

    def on_train_batch_begin(self, batch, logs=None):
        self.fedprox_loss.current_model = {layer.name: layer.get_weights() for layer in self.model.layers}


def preprocess_dataset(dataset, is_training, batch_size=1):
    """
    Apply pre-processing transformations to CIFAR10 dataset.

    Same pre-processings are used as in the Pytorch tutorial
    on CIFAR10: https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/cifar10-sim

    Training time pre-processings are (in-order):
    - Image padding with 4 pixels in "reflect" mode on each side
    - RandomCrop of 32 x 32 images
    - RandomHorizontalFlip
    - Normalize to [0, 1]: dividing pixels values by given CIFAR10 data mean & std
    - Random shuffle

    Testing/Validation time pre-processings are:
    - Normalize: dividing pixels values by 255

    Args
    ----------
    dataset: tf.data.Datset
    Tensorflow Dataset

    is_training: bool
    Boolean flag indicating if current phase is training phase.

    batch_size: int
    Batch size

    Returns
    ----------
    tf.data.Dataset
    Tensorflow Dataset with pre-processings applied.

    """
    # Values from: https://github.com/NVIDIA/NVFlare/blob/fc2bc47889b980c8de37de5528e3d07e6b1a942e/examples/advanced/cifar10/pt/learners/cifar10_model_learner.py#L147
    mean_cifar10 = tf.constant([125.3, 123.0, 113.9], dtype=tf.float32)
    std_cifar10 = tf.constant([63.0, 62.1, 66.7], dtype=tf.float32)

    if is_training:

        # Padding each dimension by 4 pixels each side
        dataset = dataset.map(
            lambda image, label: (
                tf.stack([tf.pad(tf.squeeze(t, [2]), [[4, 4], [4, 4]], mode='REFLECT')
                          for t in tf.split(image, num_or_size_splits=3, axis=2)], axis=2), label)
        )
        # Random crop of 32 x 32 x 3
        dataset = dataset.map(
            lambda image, label: (tf.image.random_crop(image, size=(32, 32, 3)), label)
        )
        # Random horizontal flip
        dataset = dataset.map(
            lambda image, label: (tf.image.random_flip_left_right(image), label)
        )
        # Normalize by dividing by given mean & std
        dataset = dataset.map(
            lambda image, label: ((tf.cast(image, tf.float32) - mean_cifar10) / std_cifar10, label)
        )
        # Random shuffle
        dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True)
        # Convert to batches.
        return dataset.batch(batch_size)

    else:

        # For validation / test only do normalization.
        return dataset.map(
            lambda image, label: ((tf.cast(image, tf.float32) - mean_cifar10) / std_cifar10, label)
        ).batch(batch_size)


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

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Use alpha-split per-site data to simulate data heteogeniety,
    # only if if train_idx_path is not None.
    #
    if args.train_idx_path != "None":

        print(f"Loading train indices from {args.train_idx_path}")
        train_idx = np.load(args.train_idx_path)
        train_images = train_images[train_idx]
        train_labels = train_labels[train_idx]

        unq, unq_cnt = np.unique(train_labels, return_counts=True)
        print(
            (f"Loaded {len(train_idx)} training indices from {args.train_idx_path} "
             "with label distribution:\nUnique labels: {unq}\nUnique Counts: {unq_cnt}")
        )

    # Convert training & testing data to datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Preprocessing
    train_ds = preprocess_dataset(train_ds, is_training=True, batch_size=args.batch_size)
    test_ds = preprocess_dataset(test_ds, is_training=False, batch_size=args.batch_size)

    model = ModerateTFNet()
    model.build(input_shape=(None, 32, 32, 3))

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs_keras", write_graph=False)]

    # Control whether FedProx is used.
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
        _, test_global_acc = model.evaluate(x=test_ds, verbose=2)
        summary_writer.add_scalar(tag="global_model_accuracy", scalar=test_global_acc, global_step=input_model.current_round)

        with tf_summary_writer.as_default():
            tf.summary.scalar("global_model_accuracy", test_global_acc, input_model.current_round)
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the {len(test_images)} test images: {test_global_acc * 100} %"
        )

        start_epoch = args.epochs*input_model.current_round
        end_epoch = start_epoch + args.epochs

        print(f"Train from epoch {start_epoch} to {end_epoch}")
        model.fit(
            x=train_ds,
            epochs=end_epoch,
            validation_data=test_ds,
            callbacks=callbacks,
            initial_epoch=start_epoch,
            validation_freq=1 #args.epochs
        )

        print("Finished Training")

        model.save_weights(PATH)

        _, test_acc = model.evaluate(x=test_ds, verbose=2)

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
