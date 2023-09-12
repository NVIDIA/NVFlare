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

from tensorflow.keras import Model, layers, losses


class TFNet(Model):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = layers.Conv2D(6, 5, activation="relu")
        self.pool = layers.MaxPooling2D((2, 2), 2)
        self.conv2 = layers.Conv2D(16, 5, activation="relu")
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation="relu")
        self.fc2 = layers.Dense(84, activation="relu")
        self.fc3 = layers.Dense(10)
        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer="sgd", loss=loss_fn, metrics=["accuracy"])
        self.build(input_shape)

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
