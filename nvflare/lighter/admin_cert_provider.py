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

from typing import Optional

from nvflare.fuel.sec.admin_cert_provider import validate_admin_cert_provider_config
from nvflare.lighter.constants import PropKey


def get_admin_cert_provider_config(admin) -> Optional[dict]:
    config = admin.get_prop(PropKey.ADMIN_CERT_PROVIDER)
    if not config:
        return None
    scope = f"admin {admin.name}.{PropKey.ADMIN_CERT_PROVIDER}"
    try:
        return validate_admin_cert_provider_config(config)
    except ValueError as ex:
        raise ValueError(f"invalid {scope}: {ex}") from ex
