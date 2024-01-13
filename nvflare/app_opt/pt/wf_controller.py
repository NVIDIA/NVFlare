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
from typing import Dict

from nvflare.app_common.workflows.wf_controller import WFController
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils.fobs import fobs


class PTWFController(WFController):
    def __init__(
        self,
        task_name: str,
        wf_class_path: str,
        wf_args: Dict,
        wf_fn_name: str = "run",
        task_timeout: int = 0,
        result_pull_interval: float = 0.2,
    ):
        super().__init__(task_name, wf_class_path, wf_args, wf_fn_name, task_timeout, result_pull_interval)

        fobs.register(TensorDecomposer)
