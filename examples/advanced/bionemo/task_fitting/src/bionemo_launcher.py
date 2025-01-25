# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import time
import torch

from nvflare.apis.dxo import MetaKey, DXO, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.security.logging import secure_format_exception
from nvflare.app_common.workflows.model_controller import ModelController
from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils.validation_utils import check_object_type

from bionemo_constants import BioNeMoConstants


class BioNeMoLauncher(Executor):
    def __init__(
        self,
        launcher_id: str = "launcher",
        task_name: str = BioNeMoConstants.TASK_INFERENCE,
        check_interval: float = 10.0
    ):
        """Run a command on the client using a launcher.

        Args:
            launcher_id (str): id of the launcher object
            task_name (str, optional): task name. Defaults to AppConstants.TASK_TRAIN.
            check_interval (float, optional): how often to check the status of the command. Defaults to 10 seconds.
        """
        super().__init__()
        self._launcher_id = launcher_id
        self._task_name = task_name
        self._check_interval = check_interval
        self.is_initialized = False        

    def _init_launcher(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        launcher: Launcher = engine.get_component(self._launcher_id)
        if launcher is None:
            raise RuntimeError(f"Launcher can not be found using {self._launcher_id}")
        check_object_type(self._launcher_id, launcher, Launcher)
        self.launcher = launcher
        self.is_initialized = True

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._task_name:        
                if not self.is_initialized:
                    self._init_launcher(fl_ctx)

                success = self._launch_script(fl_ctx)
        
                if success:
                    # Get results path from inference script arguments
                    args = self.launcher._script.split()
                    results_path = args[args.index("--results-path")+1]
                    if os.path.isfile(results_path):
                        self.log_info(fl_ctx, f"Get result info from: {results_path}")
                        results = torch.load(results_path)

                        result_shapes = {}
                        for k, v in results.items():
                            if v is not None:
                                result_shapes[k] = list(v.shape)  # turn torch Size type into a simple list for sharing with server

                        n_sequences = len(results["embeddings"])
                    else:
                        n_sequences, result_shapes = "n/a", "n/a"

                    # Prepare a DXO for our updated model. Create shareable and return
                    data_info = {BioNeMoConstants.NUMBER_SEQUENCES: n_sequences, BioNeMoConstants.RESULT_SHAPES: result_shapes}
            
                    outgoing_dxo = DXO(data_kind=BioNeMoConstants.DATA_INFO, data=data_info)
                    return outgoing_dxo.to_shareable()
                else:
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            else:
                # If unknown task name, set RC accordingly.
                return make_reply(ReturnCode.TASK_UNKNOWN)                    
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in execute: {secure_format_exception(e)}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _launch_script(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Start Executor Launcher.")

        self.launcher.initialize(fl_ctx=fl_ctx)

        success = False
        while True:
            time.sleep(self._check_interval)
            run_status = self.launcher.check_run_status(task_name=self._task_name, fl_ctx=fl_ctx)
            if run_status == LauncherRunStatus.RUNNING:
                self.log_info(fl_ctx, f"Check running command: {self.launcher._script}")
            elif run_status == LauncherRunStatus.COMPLETE_SUCCESS:
                self.log_info(fl_ctx, "Run success")
                success = True
                break
            else:
                self.log_error(fl_ctx, f"Run failed or not start: {run_status}")
                break
        self.launcher.finalize(fl_ctx=fl_ctx)
        self.log_info(fl_ctx, "Stop Executor Launcher.")
        return success
        
