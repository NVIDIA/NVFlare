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
from threading import Lock, local
from typing import Any, Dict, Optional, Union

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel

# this import is to let existing scripts import client.api
from .api_context import APIContext, ClientAPIType

global_context_lock = Lock()
context_dict = {}
default_context = None
_runtime_shutdown = False
_thread_context = local()


def _on_context_shutdown(local_ctx: APIContext) -> None:
    """Apply the process/cache consequence of one APIContext reaching shutdown."""
    global default_context
    global _runtime_shutdown

    with global_context_lock:
        if local_ctx.api_type == ClientAPIType.CELL_API:
            # A Cell context retires the trainer's process-global F3 runtime.
            _runtime_shutdown = True
        else:
            # Remove only this context: another rank/session may still be active.
            for key, cached_ctx in tuple(context_dict.items()):
                if cached_ctx is local_ctx:
                    context_dict.pop(key, None)
        if default_context is local_ctx:
            default_context = None


def get_context(ctx: Optional[APIContext] = None) -> APIContext:
    """Gets an APIContext.

    Args:
        ctx (Optional[APIContext]): The context to use. If omitted, use the context
            bound by ``init()`` to this thread, then the process default for an unbound
            helper thread. Defaults to None.

    Raises:
        RuntimeError: if can't get a valid APIContext.

    Returns:
        An APIContext.
    """
    if _runtime_shutdown:
        raise RuntimeError("Client API runtime has been shut down; start a new trainer process")
    if ctx is not None and not ctx.is_shutdown:
        return ctx
    elif ctx is not None:
        raise RuntimeError("APIContext has been shut down. Call flare.init() to create a new context.")
    bound_context = getattr(_thread_context, "context", None)
    if bound_context is not None and not bound_context.is_shutdown:
        return bound_context
    elif bound_context is not None:
        # Keep the stopped binding as a tombstone. An abandoned in-process trainer
        # thread must never fall through to a later job's process-global default.
        raise RuntimeError("Thread-bound APIContext has been shut down. Call flare.init() to create a new context.")
    elif default_context and not default_context.is_shutdown:
        return default_context
    else:
        raise RuntimeError("APIContext is None. Did you call flare.init() before using the Client API?")


def init(rank: Optional[Union[str, int]] = None, config_file: Optional[str] = None) -> APIContext:
    """Initializes NVFlare Client API environment.

    Args:
        rank (str): rank of the process for Client API control-path behavior.
            In distributed training, use the global process rank (for example torchrun's RANK),
            not the device-local rank used for GPU placement.
        config_file (str): client api configuration.

    Returns:
        APIContext: The context, also bound to the calling thread for later API calls.
    """

    # subsequent logic assumes rank is a string
    if rank is not None:
        if isinstance(rank, int):
            rank = str(rank)
        elif isinstance(rank, str):
            pass
        else:
            raise ValueError(f"rank must be a string or an integer but got {type(rank)}")

    with global_context_lock:
        global context_dict
        global default_context
        global _runtime_shutdown
        if not _runtime_shutdown:
            # CellClientAPI can stop itself on CJ SHUTDOWN; detect that process-wide close
            # before a different rank/config constructs another context.
            _runtime_shutdown = any(
                cached_ctx.api_type == ClientAPIType.CELL_API and cached_ctx.is_shutdown
                for cached_ctx in context_dict.values()
            )
        if _runtime_shutdown:
            raise RuntimeError(
                "Client API cannot be reinitialized after shutdown in the same process; " "start a new trainer process"
            )
        local_ctx = context_dict.get((rank, config_file))

        if local_ctx is None:
            local_ctx = APIContext(rank=rank, config_file=config_file)
            context_dict[(rank, config_file)] = local_ctx
            default_context = local_ctx
        elif local_ctx.is_shutdown:
            if local_ctx.api_type == ClientAPIType.CELL_API:
                raise RuntimeError(
                    "Cell Client API context has been shut down and cannot be reinitialized in the same process; "
                    "start a new trainer process"
                )
            # Non-Cell contexts are reusable across sequential jobs in the same process.
            local_ctx = APIContext(rank=rank, config_file=config_file)
            context_dict[(rank, config_file)] = local_ctx
            default_context = local_ctx
        else:
            logging.warning(
                "Warning: called init() more than once with same parameters." "The subsequence calls are ignored"
            )
        _thread_context.context = local_ctx
        return local_ctx


def receive(timeout: Optional[float] = None, ctx: Optional[APIContext] = None) -> Optional[FLModel]:
    """Receives model from NVFlare side.

    Returns:
        An FLModel received.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.receive(timeout)


def send(model: FLModel, clear_cache: bool = True, ctx: Optional[APIContext] = None) -> None:
    """Sends the model to NVFlare side.

    Args:
        model (FLModel): The FLModel object to be sent.
        clear_cache (bool): Whether to clear the cache after send.

    Raises:
        RuntimeError: If the model cannot be submitted to NVFLARE.
    """
    if not isinstance(model, FLModel):
        raise TypeError("model needs to be an instance of FLModel")
    local_ctx = get_context(ctx)
    return local_ctx.api.send(model, clear_cache)


def system_info(ctx: Optional[APIContext] = None) -> Dict:
    """Gets NVFlare system information.

    System information will be available after a valid FLModel is received.
    It does not retrieve information actively.

    Note:
        system information includes job id and site name.

    Returns:
       A dict of system information.

    """
    local_ctx = get_context(ctx)
    return local_ctx.api.system_info()


def get_config(ctx: Optional[APIContext] = None) -> Dict:
    """Gets the ClientConfig dictionary.

    Returns:
        A dict of the configuration used in Client API.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.get_config()


def get_job_id(ctx: Optional[APIContext] = None) -> str:
    """Gets job id.

    Returns:
        The current job id.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.get_job_id()


def get_site_name(ctx: Optional[APIContext] = None) -> str:
    """Gets site name.

    Returns:
        The site name of this client.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.get_site_name()


def get_task_name(ctx: Optional[APIContext] = None) -> str:
    """Gets task name.

    Returns:
        The task name.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.get_task_name()


def is_running(ctx: Optional[APIContext] = None) -> bool:
    """Returns whether the NVFlare system is up and running.

    Returns:
        True, if the system is up and running. False, otherwise.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.is_running()


def is_train(ctx: Optional[APIContext] = None) -> bool:
    """Returns whether the current task is a training task.

    Returns:
        True, if the current task is a training task. False, otherwise.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.is_train()


def is_evaluate(ctx: Optional[APIContext] = None) -> bool:
    """Returns whether the current task is an evaluate task.

    Returns:
        True, if the current task is an evaluate task. False, otherwise.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.is_evaluate()


def is_submit_model(ctx: Optional[APIContext] = None) -> bool:
    """Returns whether the current task is a submit_model task.

    Returns:
        True, if the current task is a submit_model. False, otherwise.
    """
    local_ctx = get_context(ctx)
    return local_ctx.api.is_submit_model()


def log(key: str, value: Any, data_type: AnalyticsDataType, ctx: Optional[APIContext] = None, **kwargs):
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
    local_ctx = get_context(ctx)
    return local_ctx.api.log(key, value, data_type, **kwargs)


def clear(ctx: Optional[APIContext] = None):
    """Clears the cache."""
    local_ctx = get_context(ctx)
    return local_ctx.api.clear()


def shutdown(ctx: Optional[APIContext] = None):
    """Release the explicit or calling-thread Client API context and stop its operation.

    For an external-process Cell Client API, this also retires process-global F3
    streaming services; no later Cell Client API session can run in that interpreter.
    """
    global default_context
    # Unlike other API calls, shutdown accepts stale or missing contexts so cleanup remains
    # idempotent across finally/context-manager paths.
    if ctx is not None:
        local_ctx = ctx
    else:
        # A stopped thread binding prevents late cleanup from targeting a successor job's
        # process default; unbound helper threads retain the compatibility fallback.
        local_ctx = getattr(_thread_context, "context", None)
        if local_ctx is None:
            local_ctx = default_context
    if local_ctx is None:
        return None
    try:
        return local_ctx.shutdown()
    finally:
        # APIContext invokes the same hook when shut down directly.
        _on_context_shutdown(local_ctx)
