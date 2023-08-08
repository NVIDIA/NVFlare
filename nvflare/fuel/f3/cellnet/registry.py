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
from typing import Any


class Callback:
    def __init__(self, cb, args, kwargs):
        self.cb = cb
        self.args = args
        self.kwargs = kwargs


class Registry:
    def __init__(self):
        self.reg = {}  # channel/topic => _CB

    @staticmethod
    def _item_key(channel: str, topic: str) -> str:
        return f"{channel}:{topic}"

    def set(self, channel: str, topic: str, items: Any):
        key = self._item_key(channel, topic)
        self.reg[key] = items

    def append(self, channel: str, topic: str, items: Any):
        key = self._item_key(channel, topic)
        item_list = self.reg.get(key)
        if not item_list:
            item_list = []
            self.reg[key] = item_list
        item_list.append(items)

    def find(self, channel: str, topic: str) -> Any:
        items = self.reg.get(self._item_key(channel, topic))
        if not items:
            # try topic * in channel
            items = self.reg.get(self._item_key(channel, "*"))

        if not items:
            # try topic * in channel *
            items = self.reg.get(self._item_key("*", "*"))

        return items
