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

from nvflare.app_common.abstract.fl_model import FLModel

from .task_registry import TaskRegistry


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

        if model.params is None and model.metrics is None:
            raise RuntimeError("the model to send does not have either params or metrics")

        self.submit_task(model)

    def _get_original_model(self) -> FLModel:
        if self.received_task is None:
            raise RuntimeError("no received task")
        elif self.received_task.data is None:
            raise RuntimeError("no received model")
        elif not isinstance(self.received_task.data, FLModel):
            raise RuntimeError("received_task.data is not FLModel.")
        elif self.received_task.data.params is None:
            raise RuntimeError("received_task.data.params is None.")
        return self.received_task.data
