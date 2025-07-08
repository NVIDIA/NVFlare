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
import threading


class FobsCache:

    _lock = threading.Lock()
    _items = {}

    @classmethod
    def get_item(cls, key):
        with cls._lock:
            return cls._items.get(key)

    @classmethod
    def set_item(cls, key, value):
        with cls._lock:
            cls._items[key] = value

    @classmethod
    def get_or_create(cls, key, value):
        with cls._lock:
            if key in cls._items:
                return cls._items[key]
            else:
                cls._items[key] = value
                return value

    @classmethod
    def remove(cls, key):
        with cls._lock:
            return cls._items.pop(key, None)
