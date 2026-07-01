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

from nvflare.fuel.sec.ephemeral_admin_cert import validate_ephemeral_admin_cert_config
from nvflare.lighter.constants import PropKey


def get_admin_ephemeral_cert_config(admin) -> Optional[dict]:
    config = admin.get_prop(PropKey.EPHEMERAL_ADMIN_CERT)
    if not config:
        return None
    scope = f"admin {admin.name}.{PropKey.EPHEMERAL_ADMIN_CERT}"
    try:
        return validate_ephemeral_admin_cert_config(config)
    except ValueError as ex:
        raise ValueError(f"invalid {scope}: {ex}") from ex
