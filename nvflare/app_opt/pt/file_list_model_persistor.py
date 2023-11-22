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
from typing import Dict

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.pt import PTFileModelPersistor


class PTFileListModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        exclude_vars=None,
        model=None,
        source_ckpt_file_full_name=None,
        filter_id: str = None,
        model_list={},
    ):
        """PTFileListModelPersistor

        PTFileListModelPersistor extends the functions of PTFileModelPersistor, which can
        provide a list of model names and model file locations for the PT model persistor.
        """
        self.model_list = model_list

        super().__init__(
            exclude_vars=exclude_vars,
            model=model,
            source_ckpt_file_full_name=source_ckpt_file_full_name,
            filter_id=filter_id,
        )

    def get_model_inventory(self, fl_ctx: FLContext) -> Dict[str, ModelDescriptor]:
        model_inventory = {}

        for k, v in self.model_list.items():
            if os.path.exists(v):
                model_inventory[k] = ModelDescriptor(
                    name=k,
                    location=v,
                    model_format=self.persistence_manager.get_persist_model_format(),
                    props={},
                )

        return model_inventory
