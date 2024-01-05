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

import os.path
from typing import Any

from nvflare.app_common.app_defined.model_persistor import AppDefinedModelPersistor

from .av_model import AVModel


class AVModelPersistor(AppDefinedModelPersistor):
    def __init__(self, file_path: str, output_path: str):
        AppDefinedModelPersistor.__init__(self)
        self.file_path = file_path
        if not os.path.isfile(file_path):
            raise ValueError(f"model file {file_path} does not exist")
        self.output_path = output_path

    def read_model(self) -> Any:
        self.info(f"loading model from {self.file_path}")
        return AVModel.load(self.file_path)

    def write_model(self, model_obj: Any):
        assert isinstance(model_obj, AVModel)
        model_obj.save(self.output_path)
        self.info(f"saved model in {self.output_path}")
