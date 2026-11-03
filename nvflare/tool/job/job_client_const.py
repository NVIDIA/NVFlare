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

APP_CONFIG_FILE_BASE_NAMES = ["config_fed_client", "config_fed_server"]
JOB_META_BASE_NAME = "meta"

CONFIG_FED_SERVER_CONF = "config_fed_server.conf"
CONFIG_FED_CLIENT_CONF = "config_fed_client.conf"

CONFIG_CONF = "config.conf"

DEFAULT_APP_NAME = "app"
# for consistency, meta config have a dummy app name
META_APP_NAME = "__meta__app__"
APP_CONFIG_DIR = "config"

APP_SCRIPT_KEY = "app_script"
APP_CONFIG_KEY = "app_config"
