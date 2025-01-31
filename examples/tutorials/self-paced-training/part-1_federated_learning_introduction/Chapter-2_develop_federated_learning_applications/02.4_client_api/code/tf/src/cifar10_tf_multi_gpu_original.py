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

import tensorflow as tf
from tensorflow.keras import datasets
from tf_net import TFNet

PATH = "./tf_model.weights.h5"


def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = TFNet()
        model.compile(
            optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
        )
        model.summary()

    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    print("Finished Training")

    model.save_weights(PATH)

    _, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Accuracy of the model on the 10000 test images: {test_acc * 100} %")


if __name__ == "__main__":
    main()
