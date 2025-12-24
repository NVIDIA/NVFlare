# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import tensorflow as tf
from data.cifar10_data_utils import load_cifar10_with_retry, preprocess_dataset
from model import ModerateTFNet
from tensorflow.keras import losses

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    # Load CIFAR10 data
    (train_images, train_labels), (test_images, test_labels) = load_cifar10_with_retry()

    # Convert to datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Preprocessing
    train_ds = preprocess_dataset(train_ds, is_training=True, batch_size=args.batch_size)
    test_ds = preprocess_dataset(test_ds, is_training=False, batch_size=args.batch_size)

    # Create model
    model = ModerateTFNet()
    model.build(input_shape=(None, 32, 32, 3))

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs", write_graph=False)]

    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=loss, metrics=["accuracy"])
    model.summary()

    # Train
    print(f"Training centralized model for {args.epochs} epochs")
    history = model.fit(
        x=train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        callbacks=callbacks,
        validation_freq=1,
    )

    print("Finished Training")

    # Evaluate
    _, test_acc = model.evaluate(x=test_ds, verbose=2)
    print(f"Final accuracy of the centralized model: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
