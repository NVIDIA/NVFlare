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

import os

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.utils import fobs

from .job_result_validator import FinishJobResultValidator

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


class TFModelValidator(FinishJobResultValidator):
    def __init__(self, model_file_name: str = "tf2weights.fobs"):
        super().__init__()
        if not model_file_name or os.path.basename(model_file_name) != model_file_name:
            raise ValueError("model_file_name must be a file name without directory components")
        self.model_file_name = model_file_name

    def validate_finished_results(self, job_result, client_props) -> bool:
        server_run_dir = job_result["workspace_root"]
        server_models_dir = os.path.join(server_run_dir, WorkspaceConstants.APP_PREFIX + "server")
        if not os.path.exists(server_models_dir):
            self.logger.error(f"models dir {server_models_dir} doesn't exist.")
            return False

        model_path = os.path.join(server_models_dir, self.model_file_name)
        if not os.path.isfile(model_path):
            self.logger.error(f"model_path {model_path} doesn't exist.")
            return False

        try:
            if self.model_file_name.endswith(".fobs"):
                self._validate_fobs(model_path)
            elif self.model_file_name.endswith((".h5", ".hdf5")):
                self._validate_hdf5(model_path)
            else:
                self.logger.error(f"unsupported TensorFlow model file format: {self.model_file_name}")
                return False
        except Exception as e:
            self.logger.error(f"Exception in validating TF model: {e.__str__()}")
            return False

        return True

    def _validate_fobs(self, model_path: str):
        flare_decomposers.register()
        common_decomposers.register()

        with open(model_path, "rb") as model_file:
            data = fobs.load(model_file)
        self.logger.info(f"Data loaded: {data}.")
        if not isinstance(data, dict) or "weights" not in data or "meta" not in data:
            raise ValueError("FOBS TensorFlow model must contain weights and meta")

    @staticmethod
    def _validate_hdf5(model_path: str):
        with open(model_path, "rb") as model_file:
            signature = model_file.read(len(_HDF5_SIGNATURE))
        if signature != _HDF5_SIGNATURE:
            raise ValueError("TensorFlow model is not a valid HDF5 file")
