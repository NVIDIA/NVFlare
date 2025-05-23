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

from abc import ABC, abstractmethod
from typing import Callable, List, Union

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.base_model_controller import BaseModelController


class ModelController(BaseModelController, ABC):
    def __init__(
        self,
        *args,
        persistor_id: str = AppConstants.DEFAULT_PERSISTOR_ID,
        **kwargs,
    ):
        """Model Controller API for FLModel-based Controller.

        Args:
            persistor_id (str, optional): ID of the persistor component. Defaults to AppConstants.DEFAULT_PERSISTOR_ID ("persistor").
        """
        super().__init__(*args, persistor_id, **kwargs)

    @abstractmethod
    def run(self):
        """Main `run` routine for the controller workflow."""
        raise NotImplementedError

    def send_model_and_wait(
        self,
        task_name: str = "train",
        data: FLModel = None,
        targets: Union[List[str], None] = None,
        min_responses: int = None,
        timeout: int = 0,
    ) -> List[FLModel]:
        """Send a task with data to targets and wait for results.

        Returns list of FLModel results from clients once task is completed (min_responses received or timed out).
        Results received from any clients after task is completed will be discarded.

        Args:
            task_name (str, optional): name of the task. Defaults to "train".
            data (FLModel, optional): FLModel to be sent to clients. Defaults to None.
            targets (List[str], optional): the list of target client names or None (all clients). Defaults to None.
            min_responses (int, optional): the minimum number of responses expected. If None, must receive responses from
              all clients that the task has been sent to. Defaults to None.
            timeout (int, optional): time to wait for clients to perform task. Defaults to 0 (never time out).

        Returns:
            List[FLModel]
        """
        return super().broadcast_model(
            task_name=task_name,
            data=data,
            targets=targets,
            min_responses=min_responses,
            timeout=timeout,
        )

    def send_model(
        self,
        task_name: str = "train",
        data: FLModel = None,
        targets: Union[List[str], None] = None,
        min_responses: int = None,
        timeout: int = 0,
        callback: Callable[[FLModel], None] = None,
    ) -> None:
        """Send a task with data to targets (non-blocking). Callback is called when a result is received.

        Args:
            task_name (str, optional): name of the task. Defaults to "train".
            data (FLModel, optional): FLModel to be sent to clients. Defaults to None.
            targets (List[str], optional): the list of target client names or None (all clients). Defaults to None.
            min_responses (int, optional): the minimum number of responses expected. If None, must receive responses from
              all clients that the task has been sent to. Defaults to None.
            timeout (int, optional): time to wait for clients to perform task. Defaults to 0 (never time out).
            callback (Callable[[FLModel], None], optional): callback when a result is received. Defaults to None.

        Returns:
            None
        """
        super().broadcast_model(
            task_name=task_name,
            data=data,
            targets=targets,
            min_responses=min_responses,
            timeout=timeout,
            blocking=False,
            callback=callback,
        )

    def load_model(self) -> FLModel:
        """Load initial model from persistor. If persistor is not configured, returns empty FLModel.

        Returns:
            FLModel
        """
        return super().load_model()

    def save_model(self, model: FLModel) -> None:
        """Saves model with persistor. If persistor is not configured, does not save.

        Args:
            model (FLModel): model to save.

        Returns:
            None
        """
        super().save_model(model)

    def sample_clients(self, num_clients: int = None) -> List[str]:
        """Returns a list of `num_clients` clients.

        Args:
            num_clients: number of clients to return. If None or > number available clients,
                returns all available clients. Defaults to None.

        Returns:
            A list of clients names.
        """
        return super().sample_clients(num_clients)
