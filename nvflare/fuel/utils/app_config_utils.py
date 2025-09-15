# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.fl_constant import SystemConfigs
from nvflare.fuel.utils.config_service import ConfigService


def get_positive_float_var(var_name, default):
    # use ConfigService to determine value for the specified var_name:
    #   the job config could define variable var_name;
    #   the user could define OS env var NVFLARE_VAR_NAME (the var_name turned to uppercase)
    value = ConfigService.get_float_var(name=var_name, conf=SystemConfigs.APPLICATION_CONF, default=default)
    if value is None:
        return default
    else:
        return value if value > 0.0 else default


def get_positive_int_var(var_name, default):
    value = ConfigService.get_int_var(name=var_name, conf=SystemConfigs.APPLICATION_CONF, default=default)
    if value is None:
        return default
    else:
        return value if value > 0 else default


def get_int_var(var_name, default):
    value = ConfigService.get_int_var(name=var_name, conf=SystemConfigs.APPLICATION_CONF, default=default)
    if value is None:
        return default
    else:
        return value
