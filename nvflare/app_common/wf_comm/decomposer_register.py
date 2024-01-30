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
from typing import List

from nvflare.apis.fl_component import FLComponent
from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.fuel.utils.fobs import fobs


class DecomposerRegister(FLComponent):
    def __init__(self, decomposers: List[str]):
        super(DecomposerRegister, self).__init__()
        self.decomposers = decomposers

    def register(self):
        for class_path in self.decomposers:
            d = instantiate_class(class_path, init_params=None)
            fobs.register(d)
