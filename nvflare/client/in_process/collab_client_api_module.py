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

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nvflare.client.in_process.collab_api import CollabClientAPI

_current_api: ContextVar[Optional["CollabClientAPI"]] = ContextVar("collab_client_api", default=None)


def set_api(api: "CollabClientAPI"):
    _current_api.set(api)


def get_api() -> Optional["CollabClientAPI"]:
    return _current_api.get()
