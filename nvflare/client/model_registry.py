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

from typing import Optional

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

from .config import ClientConfig
from .flare_agent import FlareAgent
from .task_registry import TaskRegistry
from .utils import DIFF_FUNCS


class ModelRegistry(TaskRegistry):
    """This class is used to remember attributes that need to be shared for a user code.

    For example, after "global_evaluate" we should remember the "metrics" value.
    And set that into the model that we want to submit after "train".

    For each user file:
        - we only need 1 model exchanger.
        - we only need to pull global model once

    """

    def __init__(self, config: ClientConfig, rank: Optional[str] = None, flare_agent: Optional[FlareAgent] = None):
        super().__init__(config, rank, flare_agent)
        self.metrics = None

    def get_model(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        """Gets a model from FLARE client.

        This method gets the task from FLARE client, and extract the `task.data` out.

        Args:
            timeout (float, optional): If specified, this call is blocked only for the specified amount of time.
                If not specified, this call is blocked forever until a task has been received or agent has been closed.

        Returns:
            None if flare agent is None; or an FLModel object if a task is available within timeout.
        """
        task = self.get_task(timeout)
        if task is not None and task.data is not None:
            if not isinstance(task.data, FLModel):
                raise RuntimeError("task.data is not FLModel.")
            return task.data
        return None

    def submit_model(self, model: FLModel) -> None:
        """Submits a model to FLARE client.

        Args:
            model (FLModel): Trained local model to be submitted.
        """
        if not self.flare_agent:
            return None
        if self.config.get_transfer_type() == "DIFF":
            exchange_format = self.config.get_exchange_format()
            diff_func = DIFF_FUNCS.get(exchange_format, None)
            if diff_func is None:
                raise RuntimeError(f"no default params diff function for {exchange_format}")
            elif self.received_task is None:
                raise RuntimeError("no received task")
            elif self.received_task.data is None:
                raise RuntimeError("no received model")
            elif not isinstance(self.received_task.data, FLModel):
                raise RuntimeError("received_task.data is not FLModel.")
            elif model.params is not None:
                if model.params_type == ParamsType.FULL:
                    try:
                        model.params = diff_func(original=self.received_task.data.params, new=model.params)
                        model.params_type = ParamsType.DIFF
                    except Exception as e:
                        raise RuntimeError(f"params diff function failed: {e}")

        if model.params is None and model.metrics is None:
            raise RuntimeError("the model to send does not have either params or metrics")

        self.submit_task(model)

    def clear(self):
        """Clears the model registry cache."""
        super().clear()
        self.metrics = None
