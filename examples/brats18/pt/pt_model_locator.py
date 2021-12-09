# Copyright (c) 2021, NVIDIA CORPORATION.
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
from typing import List

import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from nvflare.app_common.app_constant import DefaultCheckpointFileName


class PTModelLocator(ModelLocator):
    SERVER_MODEL_NAME = "server"
    SERVER_BEST_MODEL_NAME = "server_best"

    def __init__(
        self, model_dir="app_server",
        model_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_model_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL
    ):
        """A ModelLocator that provides the global and best global models.

        Args:
            model_dir: directory where global models are saved.
            model_name: name of the saved global model.
            best_model_name: name of the saved best global model.

        Returns:
            a DXO depending on the specified `model_name` in `locate_model()`.
        """
        super().__init__()

        self.model_dir = model_dir
        self.model_file_name = model_name
        self.best_model_file_name = best_model_name

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        """Returns the list of model names that should be included from server in cross site validation.add()

        Args:
            fl_ctx (FLContext): FL Context object.

        Returns:
            List[str]: List of model names.
        """
        return [PTModelLocator.SERVER_MODEL_NAME, PTModelLocator.SERVER_BEST_MODEL_NAME]

    def locate_model(self, model_name, fl_ctx: FLContext) -> DXO:
        dxo = None
        engine = fl_ctx.get_engine()

        if model_name in (PTModelLocator.SERVER_MODEL_NAME, PTModelLocator.SERVER_BEST_MODEL_NAME):
            run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
            run_dir = engine.get_workspace().get_run_dir(run_number)
            model_path = os.path.join(run_dir, self.model_dir)

            if model_name == PTModelLocator.SERVER_BEST_MODEL_NAME:
                model_load_path = os.path.join(model_path, self.best_model_file_name)
            else:
                model_load_path = os.path.join(model_path, self.model_file_name)
            model_data = None
            try:
                model_data = torch.load(model_load_path)
                self.log_info(fl_ctx, f"Loaded {model_name} model from {model_load_path}.")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load model: {e}.")

            if model_data is not None:
                mgr = PTModelPersistenceFormatManager(model_data)
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=mgr.var_dict, meta=mgr.meta)

        return dxo
