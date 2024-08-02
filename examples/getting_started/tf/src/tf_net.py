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

from tensorflow.keras import layers, models


class TFNet(models.Sequential):
    def __init__(self, input_shape=(None, 32, 32, 3)):
        super().__init__()
        self._input_shape = input_shape
        # Do not specify input as we will use delayed built only during runtime of the model
        # self.add(layers.Input(shape=(32, 32, 3)))
        self.add(layers.Conv2D(32, (3, 3), activation="relu"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.Flatten())
        self.add(layers.Dense(64, activation="relu"))
        self.add(layers.Dense(10))


class ModerateTFNet(models.Sequential):
    # Follow ModerateCNN architecture from cifar10_nets.py
    def __init__(self, input_shape=(None, 32, 32, 3)):
        super().__init__()
        self._input_shape = input_shape

        # Do not specify input as we will use delayed built only during runtime of the model
        # self.add(layers.Input(shape=(32, 32, 3)))

        # Conv Layer block 1
        self.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
        self.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
        self.add(layers.MaxPooling2D((2, 2)))

        # Conv Layer block 2
        self.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
        self.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Dropout(rate=0.05))

        # Conv Layer block 3
        self.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
        self.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Flatten())

        # FC Layer
        self.add(layers.Dropout(rate=0.1))
        self.add(layers.Dense(512, activation="relu"))
        self.add(layers.Dense(512, activation="relu"))
        self.add(layers.Dropout(rate=0.1))
        self.add(layers.Dense(10))
