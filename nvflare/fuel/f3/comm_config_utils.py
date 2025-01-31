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
from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.drivers.driver_params import DriverParams


def requires_secure(resources: dict):
    conn_sec = resources.get(DriverParams.CONNECTION_SECURITY)
    if conn_sec:
        if conn_sec == ConnectionSecurity.INSECURE:
            return False
        else:
            return True
    else:
        return resources.get(DriverParams.SECURE)
