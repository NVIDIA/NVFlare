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
import traceback
from typing import List

import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.app_constant import DefaultCheckpointFileName
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager


class SimpleModelLocator(ModelLocator):
    SERVER_MODEL_NAME = "server"
    SERVER_BEST_MODEL_NAME = "server_best"

    def __init__(
        self,
        model_dir="app_server",
        model_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_model_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
    ):
        super().__init__()

        self.model_dir = model_dir
        self.model_file_name = model_name
        self.best_model_file_name = best_model_name

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        return [SimpleModelLocator.SERVER_MODEL_NAME, SimpleModelLocator.SERVER_BEST_MODEL_NAME]

    def locate_model(self, model_name, fl_ctx: FLContext) -> DXO:
        dxo = None
        engine = fl_ctx.get_engine()
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        if model_name in self.get_model_names(fl_ctx):
            # Get run information
            run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
            run_dir = engine.get_workspace().get_run_dir(run_number)
            model_path = os.path.join(run_dir, self.model_dir)

            # Generate model path
            if model_name == SimpleModelLocator.SERVER_BEST_MODEL_NAME:
                model_load_path = os.path.join(model_path, self.best_model_file_name)
            else:
                model_load_path = os.path.join(model_path, self.model_file_name)

            # Load checkpoint
            model_data = None
            try:
                checkpoint = torch.load(model_load_path, map_location="cpu")
                model_data = checkpoint["model"]
                for var_name in model_data:
                    w = model_data[var_name]
                    if isinstance(w, torch.Tensor):
                        model_data[var_name] = w.numpy()
            except:
                self.log_error(fl_ctx, traceback.format_exc())

            if model_data is not None:
                mgr = PTModelPersistenceFormatManager(model_data)
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=mgr.var_dict, meta=mgr.meta)

        return dxo
