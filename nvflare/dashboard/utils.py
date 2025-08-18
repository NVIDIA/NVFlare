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
import os


class EnvVar:
    WEB_ROOT = "NVFL_WEB_ROOT"
    SECRET_KEY = "SECRET_KEY"
    DATABASE_URL = "DATABASE_URL"
    DASHBOARD_STATIC_FOLDER = "NVFL_DASHBOARD_STATIC_FOLDER"
    CREDENTIAL = "NVFL_CREDENTIAL"
    WEB_PORT = "NVFL_WEB_PORT"
    DASHBOARD_PP = "NVFL_DASHBOARD_PP"


def get_web_root():
    return os.environ.get(EnvVar.WEB_ROOT, "/var/tmp/nvflare/dashboard")
