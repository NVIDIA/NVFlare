# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""TensorFlow model definition for CIFAR-10."""

from tensorflow.keras import layers, models


class TFNet(models.Sequential):
    """Simple CNN for CIFAR-10 classification."""

    def __init__(self, input_shape=(None, 32, 32, 3)):
        super().__init__()
        # refer to nvflare/app_opt/tf/model_persistor.py
        self._input_shape = input_shape
        self.add(layers.Conv2D(32, (3, 3), activation="relu"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.Flatten())
        self.add(layers.Dense(64, activation="relu"))
        self.add(layers.Dense(10))
