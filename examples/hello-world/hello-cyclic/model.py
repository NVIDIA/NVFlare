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

from tensorflow.keras import layers, models


class Net(models.Sequential):
    def __init__(self, input_shape=(None, 28, 28)):
        super().__init__()
        self._input_shape = input_shape
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation="relu"))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(10))
