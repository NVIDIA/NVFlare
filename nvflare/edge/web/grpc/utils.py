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
import json
from typing import Optional

from .constants import NONE_DATA
from .edge_api_pb2 import Reply


def to_bytes(data: Optional[dict]) -> bytes:
    if not data:
        return NONE_DATA
    str_data = json.dumps(data)
    return str_data.encode("utf-8")


def make_reply(status: str, payload: Optional[dict] = None):
    return Reply(
        status=status,
        payload=to_bytes(payload),
    )
