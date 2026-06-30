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

from typing import Mapping, Optional

from nvflare.lighter.constants import PropKey


def get_admin_ephemeral_cert_config(admin) -> Optional[dict]:
    config = admin.get_prop(PropKey.EPHEMERAL_ADMIN_CERT)
    if not config:
        return None
    scope = f"admin {admin.name}.{PropKey.EPHEMERAL_ADMIN_CERT}"
    if not isinstance(config, Mapping):
        raise ValueError(f"{scope} must be a mapping but got {type(config)}")

    result = dict(config)
    provider = result.get("provider")
    if not provider:
        raise ValueError(f"{scope}.provider is required")

    provider_config = result.get("provider_config") or {}
    if not isinstance(provider_config, Mapping):
        raise ValueError(f"{scope}.provider_config must be a mapping but got {type(provider_config)}")
    result["provider_config"] = dict(provider_config)

    if "renewal_window" in result:
        try:
            renewal_window = float(result["renewal_window"])
        except (TypeError, ValueError) as ex:
            raise ValueError(f"{scope}.renewal_window must be a number") from ex
        if renewal_window <= 0.0:
            raise ValueError(f"{scope}.renewal_window must be greater than zero")
    return result
