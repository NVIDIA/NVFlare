# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.utils.stats_utils import new_time_pool, new_message_size_pool, CounterPool, HistPool


class StatsPoolManager:

    POOL_TYPE_TIME = "time"
    POOL_TYPE_MSG_SIZE = "size"

    pools = {}     # name => pool

    @classmethod
    def add_time_hist_pool(cls, name: str, description: str):
        if name in cls.pools:
            raise ValueError(f"pool '{name}' is already defined")

        p = new_time_pool(name, description)
        cls.pools[name] = p
        return p

    @classmethod
    def add_msg_size_pool(cls, name: str, description: str):
        if name in cls.pools:
            raise ValueError(f"pool '{name}' is already defined")

        p = new_message_size_pool(name, description)
        cls.pools[name] = p
        return p

    @classmethod
    def add_counter_pool(cls, name: str, description: str, counter_names: list):
        if name in cls.pools:
            raise ValueError(f"pool '{name}' is already defined")

        p = CounterPool(name, description, counter_names)
        cls.pools[name] = p
        return p

    @classmethod
    def get_pool(cls, name: str):
        return cls.pools.get(name)

    @classmethod
    def delete_pool(cls, name: str):
        return cls.pools.pop(name, None)

    @classmethod
    def get_table(cls):
        headers = ['pool', 'type', 'description']
        rows = []
        for k, v in cls.pools.items():
            r = [v.name]
            if isinstance(v, HistPool):
                t = 'hist'
            elif isinstance(v, CounterPool):
                t = 'counter'
            else:
                t = '?'
            r.append(t)
            r.append(v.description)
            rows.append(r)
        return headers, rows

    @classmethod
    def to_dict(cls):
        result = {}
        for k, v in cls.pools.items():
            if isinstance(v, HistPool):
                t = 'hist'
            elif isinstance(v, CounterPool):
                t = 'counter'
            else:
                raise ValueError(f"unknown type of pool '{k}'")
            result[k] = {
                'type': t,
                'pool': v.to_dict()
            }
        return result
