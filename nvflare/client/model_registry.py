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

from .config import TransferType
from .task_registry import TaskRegistry
from .utils import DIFF_FUNCS


class ModelRegistry(TaskRegistry):
    """Gets and submits FLModel."""

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
        if self.config.get_transfer_type() == TransferType.DIFF:
            model = self._prepare_param_diff(model)

        if model.params is None and model.metrics is None:
            raise RuntimeError("the model to send does not have either params or metrics")

        self.submit_task(model)

    def release_params(self, sent_model: FLModel) -> None:
        """Release large parameter arrays after serialization is complete.

        Called by the API layer (conditioned on clear_cache) after submit_model()
        has serialized and sent the model via pipe. Nulls both the sent model's
        params and the received model's params — neither is needed on the client
        after flare.send() returns.

        After this call, sent_model.params and the received model's params will
        be None. Callers must not access these fields after flare.send().

        Args:
            sent_model: The FLModel that was just submitted.
        """
        sent_model.params = None
        sent_model.optimizer_params = None
        if self.received_task and self.received_task.data:
            self.received_task.data.params = None
            self.received_task.data.optimizer_params = None

    def _prepare_param_diff(self, model: FLModel) -> FLModel:
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

        return model
