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
from tf_net import TFNet

import nvflare.client as flare

WEIGHTS_PATH = "./tf_model.weights.h5"


def main():
    flare.init()

    sys_info = flare.system_info()
    print(f"system info is: {sys_info}", flush=True)

    model = TFNet()
    model.build(input_shape=(None, 28, 28))
    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    model.summary()

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = (
        train_images / 255.0,
        test_images / 255.0,
    )

    # simulate separate datasets for each client by dividing MNIST dataset in half
    client_name = sys_info["site_name"]
    if client_name == "site-1":
        train_images = train_images[: len(train_images) // 2]
        train_labels = train_labels[: len(train_labels) // 2]
        test_images = test_images[: len(test_images) // 2]
        test_labels = test_labels[: len(test_labels) // 2]
    elif client_name == "site-2":
        train_images = train_images[len(train_images) // 2 :]
        train_labels = train_labels[len(train_labels) // 2 :]
        test_images = test_images[len(test_images) // 2 :]
        test_labels = test_labels[len(test_labels) // 2 :]

    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}")

        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)

        _, test_global_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the test images: {test_global_acc * 100} %"
        )

        # training
        model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

        print("Finished Training")

        model.save_weights(WEIGHTS_PATH)

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}", flush=True)
        print(f"finished round: {input_model.current_round}", flush=True)

        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers},
            params_type="FULL",
            metrics={"accuracy": test_global_acc},
            current_round=input_model.current_round,
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
