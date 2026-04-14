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

from nvflare.apis.fl_constant import FLMetaKey

SYS_ATTRS = (FLMetaKey.JOB_ID, FLMetaKey.SITE_NAME)
CLIENT_API_CONFIG = "client_api_config.json"

# Configuration key for overriding external_pre_init_timeout in ClientAPILauncherExecutor
# Jobs can set this via add_client_config() to allow more time for heavy library imports
EXTERNAL_PRE_INIT_TIMEOUT = "EXTERNAL_PRE_INIT_TIMEOUT"

# Configuration key for overriding peer_read_timeout in ClientAPILauncherExecutor.
# peer_read_timeout controls how long CJ waits for the subprocess to acknowledge
# receiving the task pipe message.  It must be long enough for the subprocess to call
# flare.get_task() after startup.  Jobs can increase it via:
#   recipe.add_client_config({"PEER_READ_TIMEOUT": 1800})
# Keeping this in the same config file as submit_result_timeout makes it easy
# to review and tune both timeouts together.
PEER_READ_TIMEOUT = "PEER_READ_TIMEOUT"
