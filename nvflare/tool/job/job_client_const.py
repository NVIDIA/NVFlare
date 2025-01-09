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

JOB_INFO_DESC_KEY = "description"
JOB_INFO_DESC = "Description"
JOB_INFO_CONTROLLER_TYPE_KEY = "controller_type"
JOB_INFO_CONTROLLER_TYPE = "Controller Type"
JOB_INFO_EXECUTION_API_TYPE_KEY = "execution_api_type"
JOB_INFO_EXECUTION_API_TYPE = "Execution API Type"
JOB_TEMPLATES = "job_templates"
JOB_TEMPLATE = "job_template"

JOB_TEMPLATE_CONF = "job_templates.conf"
JOB_INFO_CONF = "info.conf"
JOB_INFO_MD = "info.md"

JOB_INFO_KEYS = [JOB_INFO_DESC_KEY, JOB_INFO_CONTROLLER_TYPE_KEY, JOB_INFO_EXECUTION_API_TYPE_KEY]
CONFIG_FILE_BASE_NAME_WO_EXTS = ["config_fed_client", "config_fed_server", "meta"]

APP_CONFIG_FILE_BASE_NAMES = ["config_fed_client", "config_fed_server"]
JOB_META_BASE_NAME = "meta"

CONFIG_FED_SERVER_CONF = "config_fed_server.conf"
CONFIG_FED_CLIENT_CONF = "config_fed_client.conf"

JOB_CONFIG_FILE_NAME = "file_name"
JOB_CONFIG_VAR_NAME = "var_name"
JOB_CONFIG_VAR_VALUE = "value"
JOB_CONFIG_COMP_NAME = "component"
JOB_TEMPLATE_NAME = "name"

CONFIG_CONF = "config.conf"

DEFAULT_APP_NAME = "app"
# for consistency, meta config have a dummy app name
META_APP_NAME = "__meta__app__"
APP_CONFIG_DIR = "config"

APP_SCRIPT_KEY = "app_script"
APP_CONFIG_KEY = "app_config"
TEMPLATES_KEY = "templates"
