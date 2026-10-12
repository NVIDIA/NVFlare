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

"""Client API metadata shared by resident and disposable in-process scripts."""

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.client.config import ConfigKey


def prepare_task_metadata(context, site_name, job_id, task_name):
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
