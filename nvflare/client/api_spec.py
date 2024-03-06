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
from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel

CLIENT_API_KEY = "CLIENT_API"
CLIENT_API_TYPE_KEY = "CLIENT_API_TYPE"


class APISpec(ABC):
    @abstractmethod
    def init(self, rank: Optional[str] = None):
        """Initializes NVFlare Client API environment.

        Args:
            rank (str): local rank of the process.
                It is only useful when the training script has multiple worker processes. (for example multi GPU)

        Returns:
            None

        Example:

            .. code-block:: python

                nvflare.client.init()

        """
        pass

    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        """Receives model from NVFlare side.

        Returns:
            An FLModel received.

        Example:

            .. code-block:: python

                nvflare.client.receive()

        """
        pass

    @abstractmethod
    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        """Sends the model to NVFlare side.

        Args:
            fl_model (FLModel): Sends a FLModel object.
            clear_cache (bool): clear cache after send.

        Example:

            .. code-block:: python

                nvflare.client.send(fl_model=FLModel(...))

        """
        pass

    @abstractmethod
    def system_info(self) -> Dict:
        """Gets NVFlare system information.

        System information will be available after a valid FLModel is received.
        It does not retrieve information actively.

        Note:
            system information includes job id and site name.

        Returns:
        A dict of system information.

        Example:

            .. code-block:: python

                sys_info = nvflare.client.system_info()

        """
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        """Gets the ClientConfig dictionary.

        Returns:
            A dict of the configuration used in Client API.

        Example:

            .. code-block:: python

                config = nvflare.client.get_config()

        """
        pass

    @abstractmethod
    def get_job_id(self) -> str:
        """Gets job id.

        Returns:
            The current job id.

        Example:

            .. code-block:: python

                job_id = nvflare.client.get_job_id()

        """
        pass

    @abstractmethod
    def get_site_name(self) -> str:
        """Gets site name.

        Returns:
            The site name of this client.

        Example:

            .. code-block:: python

                site_name = nvflare.client.get_site_name()

        """
        pass

    @abstractmethod
    def get_task_name(self) -> str:
        """Gets task name.

        Returns:
            The task name.

        Example:

            .. code-block:: python

                task_name = nvflare.client.get_task_name()

        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Returns whether the NVFlare system is up and running.

        Returns:
            True, if the system is up and running. False, otherwise.

        Example:

            .. code-block:: python

                while nvflare.client.is_running():
                    # receive model, perform task, send model, etc.
                    ...

        """
        pass

    @abstractmethod
    def is_train(self) -> bool:
        """Returns whether the current task is a training task.

        Returns:
            True, if the current task is a training task. False, otherwise.

        Example:

            .. code-block:: python

                if nvflare.client.is_train():
                # perform train task on received model
                    ...

        """
        pass

    @abstractmethod
    def is_evaluate(self) -> bool:
        """Returns whether the current task is an evaluate task.

        Returns:
            True, if the current task is an evaluate task. False, otherwise.

        Example:

            .. code-block:: python

                if nvflare.client.is_evaluate():
                # perform evaluate task on received model
                    ...

        """
        pass

    @abstractmethod
    def is_submit_model(self) -> bool:
        """Returns whether the current task is a submit_model task.

        Returns:
            True, if the current task is a submit_model. False, otherwise.

        Example:

            .. code-block:: python

                if nvflare.client.is_submit_model():
                # perform submit_model task to obtain the best local model
                    ...

        """
        pass

    @abstractmethod
    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Logs a key value pair.

        We suggest users use the high-level APIs in nvflare/client/tracking.py

        Args:
            key (str): key string.
            value (Any): value to log.
            data_type (AnalyticsDataType): the data type of the "value".
            kwargs: additional arguments to be included.

        Returns:
            whether the key value pair is logged successfully

        Example:

            .. code-block:: python

                log(
                    key=tag,
                    value=scalar,
                    data_type=AnalyticsDataType.SCALAR,
                    global_step=global_step,
                    writer=LogWriterName.TORCH_TB,
                    **kwargs,
                )

        """
        pass

    @abstractmethod
    def clear(self):
        """Clears the cache.

        Example:

            .. code-block:: python

                nvflare.client.clear()

        """
        pass
