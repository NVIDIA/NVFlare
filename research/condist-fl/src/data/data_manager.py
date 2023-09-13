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

from typing import Dict, Optional

from .data_loader import create_data_loader
from .dataset import create_dataset


class DataManager(object):
    def __init__(self, app_root: str, config: Dict):
        self.app_root = app_root
        self.config = config

        self._dataset = {}
        self._data_loader = {}

    def _build_dataset(self, stage: str):
        if stage == "train":
            mode = "train"
            split = "training"
        elif stage == "validate":
            mode = "validate"
            split = "validation"
        elif stage == "test":
            mode = "validate"
            split = "testing"
        else:
            raise ValueError(f"Unknown stage {stage} for dataset")
        return create_dataset(self.app_root, self.config, split, mode)

    def _build_data_loader(self, stage: str):
        ds = self._dataset.get(stage)
        if stage == "train":
            dl = create_data_loader(
                ds,
                batch_size=self.config["data_loader"].get("batch_size", 1),
                num_workers=self.config["data_loader"].get("num_workers", 0),
                shuffle=True,
            )
        else:
            dl = create_data_loader(ds, batch_size=1, num_workers=self.config["data_loader"].get("num_workers", 0))
        return dl

    def setup(self, stage: Optional[str] = None):
        if stage is None:
            for s in ["train", "validate", "test"]:
                self._dataset[s] = self._build_dataset(s)
                self._data_loader[s] = _build_data_loader(s)
        elif stage in ["train", "validate", "test"]:
            self._dataset[stage] = self._build_dataset(stage)
            self._data_loader[stage] = self._build_data_loader(stage)

    def get_dataset(self, stage: str):
        return self._dataset.get(stage, None)

    def get_data_loader(self, stage: str):
        return self._data_loader.get(stage, None)

    def teardown(self):
        self._dataset = {}
        self._data_loader = {}
