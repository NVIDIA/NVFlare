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

import threading

from nvflare.fuel.utils.stats_utils import CounterPool, HistPool, new_message_size_pool, new_time_pool


class StatsPoolManager:

    lock = threading.Lock()
    pools = {}  # name => pool

    @classmethod
    def _check_name(cls, name, scope):
        name = name.lower()
        if name not in cls.pools:
            return name
        if scope:
            name = f"{name}@{scope}"
            if name not in cls.pools:
                return name
        raise ValueError(f"pool '{name}' is already defined")

    @classmethod
    def add_time_hist_pool(cls, name: str, description: str, marks=None, scope=None):
        name = cls._check_name(name, scope)
        p = new_time_pool(name, description, marks)
        cls.pools[name] = p
        return p

    @classmethod
    def add_msg_size_pool(cls, name: str, description: str, marks=None, scope=None):
        name = cls._check_name(name, scope)
        p = new_message_size_pool(name, description, marks)
        cls.pools[name] = p
        return p

    @classmethod
    def add_counter_pool(cls, name: str, description: str, counter_names: list, scope=None):
        name = cls._check_name(name, scope)
        p = CounterPool(name, description, counter_names)
        cls.pools[name] = p
        return p

    @classmethod
    def get_pool(cls, name: str):
        name = name.lower()
        return cls.pools.get(name)

    @classmethod
    def delete_pool(cls, name: str):
        with cls.lock:
            name = name.lower()
            return cls.pools.pop(name, None)

    @classmethod
    def get_table(cls):
        with cls.lock:
            headers = ["pool", "type", "description"]
            rows = []
            for k in sorted(cls.pools.keys()):
                v = cls.pools[k]
                r = [v.name]
                if isinstance(v, HistPool):
                    t = "hist"
                elif isinstance(v, CounterPool):
                    t = "counter"
                else:
                    t = "?"
                r.append(t)
                r.append(v.description)
                rows.append(r)
            return headers, rows

    @classmethod
    def to_dict(cls):
        with cls.lock:
            result = {}
            for k in sorted(cls.pools.keys()):
                v = cls.pools[k]
                if isinstance(v, HistPool):
                    t = "hist"
                elif isinstance(v, CounterPool):
                    t = "counter"
                else:
                    raise ValueError(f"unknown type of pool '{k}'")
                result[k] = {"type": t, "pool": v.to_dict()}
            return result

    @classmethod
    def from_dict(cls, d: dict):
        cls.pools = {}
        for k, v in d.items():
            t = v.get("type")
            if not t:
                raise ValueError("missing pool type")
            pd = v.get("pool")
            if not pd:
                raise ValueError("missing pool data")

            if t == "hist":
                p = HistPool.from_dict(pd)
            elif t == "counter":
                p = CounterPool.from_dict(pd)
            else:
                raise ValueError(f"invalid pool type {t}")
            cls.pools[k] = p
