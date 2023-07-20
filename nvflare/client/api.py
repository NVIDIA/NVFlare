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

import copy
import os
from typing import Any, Dict, Optional, Tuple

from nvflare.app_common.model_exchange.file_pipe_model_exchanger import FilePipeModelExchanger
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import

from .config import ClientConfig
from .constants import ModelExchangeFormat
from .model_cache import Cache

PROCESS_CACHE: Dict[int, Cache] = {}


# TODO: some other helper methods:
#   - get_total_rounds()
#   - get_job_id()


def init(config: str = "config/config_exchange.json"):
    pid = os.getpid()
    if pid in PROCESS_CACHE:
        raise RuntimeError("Can't call init twice.")

    if not os.path.exists(config):
        raise RuntimeError(f"Missing config file {config}.")

    client_config = ClientConfig(config_file=config)
    if client_config.get_exchange_format() == ModelExchangeFormat.PYTORCH:
        tensor_decomposer, ok = optional_import(module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer")
        if ok:
            fobs.register(tensor_decomposer)
        else:
            raise RuntimeError(f"Can't import TensorDecomposer for format: {ModelExchangeFormat.PYTORCH}")

    mdx = FilePipeModelExchanger(data_exchange_path=client_config.get_exchange_path())
    PROCESS_CACHE[pid] = Cache(mdx, client_config)


def receive_model() -> Tuple[Any, Dict]:
    """Receives model from NVFlare side.

    Returns:
        A tuple of model, metadata received.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    model = copy.deepcopy(cache.input_model.params)
    meta = cache.meta
    return model, meta


def submit_metrics(metrics: Dict, meta: Optional[Dict] = None) -> None:
    """Submits metrics.

    Args:
        metrics (Dict): metrics to be submitted.
        meta: the metadata to be sumbmitted along with the metrics.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    if meta is not None:
        cache.meta.update(meta)
    cache.metrics = metrics


def submit_model(model: Any, meta: Optional[Dict] = None) -> None:
    """Submits model.

    Args:
        model: model to be submitted.
        meta: the metadata to be sumbmitted along with the model.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    if meta is not None:
        cache.meta.update(meta)

    fl_model = cache.construct_fl_model(params=model)

    cache.model_exchanger.submit_model(model=fl_model)
    cache.model_exchanger.finalize(close_pipe=False)
    PROCESS_CACHE.pop(pid)


def get_sys_meta() -> Dict:
    """Gets NVFlare system metadata.

    Returns:
       A dict of system metadata.
    """
    pid = os.getpid()
    if pid not in PROCESS_CACHE:
        raise RuntimeError("needs to call init method first")
    cache = PROCESS_CACHE[pid]
    return cache.sys_meta
