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

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.fuel.data_event.data_bus import DataBus

from .api_spec import CLIENT_API_KEY, CLIENT_API_TYPE_KEY, APISpec
from .ex_process.api import ExProcessClientAPI


class ClientAPIType(Enum):
    IN_PROCESS_API = "IN_PROCESS_API"
    EX_PROCESS_API = "EX_PROCESS_API"


client_api: Optional[APISpec] = None
data_bus = DataBus()


def init(rank: Optional[str] = None):
    """Initializes NVFlare Client API environment.

    Args:
        rank (str): local rank of the process.
            It is only useful when the training script has multiple worker processes. (for example multi GPU)

    Returns:
        None
    """
    api_type_name = os.environ.get(CLIENT_API_TYPE_KEY, ClientAPIType.IN_PROCESS_API.value)
    api_type = ClientAPIType(api_type_name)
    global client_api
    if client_api is None:
        if api_type == ClientAPIType.IN_PROCESS_API:
            client_api = data_bus.get_data(CLIENT_API_KEY)
        else:
            client_api = ExProcessClientAPI()
        client_api.init(rank=rank)
    else:
        logging.warning("Warning: called init() more than once. The subsequence calls are ignored")


def receive(timeout: Optional[float] = None) -> Optional[FLModel]:
    """Receives model from NVFlare side.

    Returns:
        An FLModel received.
    """
    global client_api
    return client_api.receive(timeout)


def send(model: FLModel, clear_cache: bool = True) -> None:
    """Sends the model to NVFlare side.

    Args:
        model (FLModel): Sends a FLModel object.
        clear_cache: clear cache after send
    """
    if not isinstance(model, FLModel):
        raise TypeError("model needs to be an instance of FLModel")
    global client_api
    return client_api.send(model, clear_cache)


def system_info() -> Dict:
    """Gets NVFlare system information.

    System information will be available after a valid FLModel is received.
    It does not retrieve information actively.

    Note:
        system information includes job id and site name.

    Returns:
       A dict of system information.

    """
    global client_api
    return client_api.system_info()


def get_config() -> Dict:
    """Gets the ClientConfig dictionary.

    Returns:
        A dict of the configuration used in Client API.
    """
    global client_api
    return client_api.get_config()


def get_job_id() -> str:
    """Gets job id.

    Returns:
        The current job id.
    """
    global client_api
    return client_api.get_job_id()


def get_site_name() -> str:
    """Gets site name.

    Returns:
        The site name of this client.
    """
    global client_api
    return client_api.get_site_name()


def get_task_name() -> str:
    """Gets task name.

    Returns:
        The task name.
    """
    global client_api
    return client_api.get_task_name()


def is_running() -> bool:
    """Returns whether the NVFlare system is up and running.

    Returns:
        True, if the system is up and running. False, otherwise.
    """
    global client_api
    return client_api.is_running()


def is_train() -> bool:
    """Returns whether the current task is a training task.

    Returns:
        True, if the current task is a training task. False, otherwise.
    """
    global client_api
    return client_api.is_train()


def is_evaluate() -> bool:
    """Returns whether the current task is an evaluate task.

    Returns:
        True, if the current task is an evaluate task. False, otherwise.
    """
    global client_api
    return client_api.is_evaluate()


def is_submit_model() -> bool:
    """Returns whether the current task is a submit_model task.

    Returns:
        True, if the current task is a submit_model. False, otherwise.
    """
    global client_api
    return client_api.is_submit_model()


def log(key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
    """Logs a key value pair.

    We suggest users use the high-level APIs in nvflare/client/tracking.py

    Args:
        key (str): key string.
        value (Any): value to log.
        data_type (AnalyticsDataType): the data type of the "value".
        kwargs: additional arguments to be included.

    Returns:
        whether the key value pair is logged successfully
    """
    global client_api
    return client_api.log(key, value, data_type, **kwargs)


def clear():
    """Clears the cache."""
    global client_api
    return client_api.clear()
