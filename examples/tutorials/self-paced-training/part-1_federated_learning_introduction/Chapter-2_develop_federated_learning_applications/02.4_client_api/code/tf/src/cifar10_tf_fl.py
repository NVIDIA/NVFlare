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

# (1) import nvflare client API
import nvflare.client as flare

PATH = "./tf_model.weights.h5"


def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = TFNet()
    model.build(input_shape=(None, 32, 32, 3))
    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    model.summary()

    # (2) initializes NVFlare client API
    flare.init()

    # (3) gets FLModel from NVFlare
    while flare.is_running():
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
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the 10000 test images: {test_global_acc * 100} %"
        )

        model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

        print("Finished Training")

        model.save_weights(PATH)

        _, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f"Accuracy of the model on the 10000 test images: {test_acc * 100} %")

        # (6) construct trained FL model (A dict of {layer name: layer weights} from the keras model)
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers}, metrics={"accuracy": test_global_acc}
        )
        # (7) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
