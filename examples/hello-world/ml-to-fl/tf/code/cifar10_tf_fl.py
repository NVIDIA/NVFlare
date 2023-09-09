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
from tf_utils import get_flat_weights, load_flat_weights

# (1) import nvflare client API
import nvflare.client as flare

# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = "cuda:0"
PATH = "./cifar_net.keras"


def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    net = TFNet()
    net.compile(
        optimizer="sgd", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    _ = net(tf.keras.Input(shape=(32, 32, 3)))
    net.summary()

    # (2) initializes NVFlare client API
    flare.init()

    # (3) gets FLModel from NVFlare
    input_model = flare.receive()

    # (4) loads model from NVFlare
    load_flat_weights(net, input_model.params)

    # (5) evaluate aggregated/received model
    _, test_global_acc = net.evaluate(test_images, test_labels, verbose=2)
    print(f"Accuracy of the received network on the 10000 test images: {test_global_acc} %")

    net.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

    print("Finished Training")

    net.save(PATH)

    _, test_acc = net.evaluate(test_images, test_labels, verbose=2)
    print(f"Accuracy of the network on the 10000 test images: {test_acc} %")

    # (6) construct trained FL model
    output_model = flare.FLModel(params=get_flat_weights(net), metrics={"accuracy": test_global_acc})
    # (7) send model back to NVFlare
    flare.send(output_model)


if __name__ == "__main__":
    main()
