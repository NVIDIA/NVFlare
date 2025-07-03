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
import uuid

_NAME_PREFIX = "_admin_"
_NAME_PREFIX_LEN = len(_NAME_PREFIX)


def new_admin_client_name():
    return f"{_NAME_PREFIX}{uuid.uuid4()}"


def _is_valid_uuid(data: str) -> bool:
    try:
        val = uuid.UUID(data, version=4)
    except ValueError:
        return False

    # If the jid string is a valid hex code, but an invalid uuid4,the UUID.__init__ will convert it to a
    # valid uuid4. This is bad for validation purposes.
    return val.hex == data.replace("-", "")


def is_valid_admin_client_name(name: str) -> bool:
    if not isinstance(name, str):
        return False

    if not name.startswith(_NAME_PREFIX):
        return False

    return _is_valid_uuid(name[_NAME_PREFIX_LEN:])
