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

import json


class AVModel:
    def __init__(self, meta, frozen_layers, free_layers):
        self.meta = meta
        self.frozen_layers = frozen_layers
        self.free_layers = free_layers

    def save(self, file_name: str):
        json.dump(
            {"free": self.free_layers, "frozen": self.frozen_layers, "meta": self.meta},
            open(file_name, "w"),
            indent=4,
            sort_keys=True,
        )

    @classmethod
    def load(cls, file_name: str):
        data = json.load(open(file_name, "r"))
        return cls(meta=data.get("meta", {}), frozen_layers=data.get("frozen", {}), free_layers=data.get("free", {}))
