# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared local Client API setup; execution lifetime stays with each backend."""

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.config import ConfigKey
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.security.logging import secure_format_traceback


def prepare_task_metadata(context, site_name, job_id, task_name):
    """Transport the declared model exchange contract without converting it."""
    return {
        FLMetaKey.SITE_NAME: site_name,
        FLMetaKey.JOB_ID: job_id,
        ConfigKey.TASK_NAME: task_name,
        ConfigKey.TASK_EXCHANGE: {
            ConfigKey.TRAIN_WITH_EVAL: context.train_with_evaluation,
            ConfigKey.EXCHANGE_FORMAT: context.params_exchange_format,
            ConfigKey.SERVER_EXPECTED_FORMAT: context.server_expected_format,
            ConfigKey.TRANSFER_TYPE: context.params_transfer_type,
            ConfigKey.TRAIN_TASK_NAME: context.train_task_name,
            ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
        },
    }


def create_script_binding(context, metadata, custom_dir, api_factory, logger):
    """Initialize an API and runner without starting a script or binding the bus.

    The caller chooses its API transport and when/how the script runs. Partial
    API setup is unwound here; backend-owned callbacks remain caller-owned.
    """
    register_framework_decomposers(context.params_exchange_format, context.server_expected_format, logger)
    api = api_factory(metadata)
    try:
        api.init()
        if context.memory_gc_rounds > 0:
            api.configure_memory_management(
                gc_rounds=context.memory_gc_rounds, cuda_empty_cache=context.cuda_empty_cache
            )
        runner = TaskScriptRunner(
            custom_dir=custom_dir, script_path=context.task_script_path, script_args=context.task_script_args
        )
    except BaseException:
        close_script_api(api, on_error=logger.error)
        raise
    return api, runner


def close_script_api(api, data_bus=None, on_error=None):
    """Close an API and clear only the bus entry it still owns.

    Disposable backends propagate cleanup errors; job backends supply their
    logger to keep cleanup best-effort. Clearing is attempted even if closing
    fails, and never removes another backend's binding.
    """
    if api is None:
        return
    try:
        try:
            api.close()
        except Exception:
            if on_error is None:
                raise
            on_error(secure_format_traceback())
    finally:
        try:
            if data_bus is not None and data_bus.get_data(CLIENT_API_KEY) is api:
                data_bus.put_data(CLIENT_API_KEY, None)
        except Exception:
            if on_error is None:
                raise
            on_error(secure_format_traceback())
