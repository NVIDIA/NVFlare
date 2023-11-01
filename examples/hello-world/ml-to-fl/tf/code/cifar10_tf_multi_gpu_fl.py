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

# (2) import how to load / dump flat weights
from nvflare.app_opt.tf.utils import get_flat_weights, load_flat_weights

PATH = "./tf_model.ckpt"


def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = TFNet(input_shape=(None, 32, 32, 3))
        model.summary()

    # (3) initializes NVFlare client API
    flare.init()

    # (4) gets FLModel from NVFlare
    for input_model in flare.receive_global_model():
        print(f"current_round={input_model.current_round}")

        # (optional) print system info
        system_info = flare.system_info()
        print(f"NVFlare system info: {system_info}")

        # (5) loads model from NVFlare
        load_flat_weights(model, input_model.params)

        # (6) evaluate aggregated/received model
        _, test_global_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the 10000 test images: {test_global_acc * 100} %"
        )

        model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

        print("Finished Training")

        model.save_weights(PATH)

        _, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f"Accuracy of the model on the 10000 test images: {test_acc * 100} %")

        # (7) construct trained FL model
        output_model = flare.FLModel(params=get_flat_weights(model), metrics={"accuracy": test_global_acc})
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
