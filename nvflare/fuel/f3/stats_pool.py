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

import csv
import json
import threading

from nvflare.fuel.utils.stats_utils import CounterPool, HistPool, new_message_size_pool, new_time_pool


class StatsPoolManager:

    _ROW_TYPE_POOL = "P"
    _ROW_TYPE_CAT = "C"
    _ROW_TYPE_REC = "R"

    _CONFIG_KEY_SAVE_POOLS = "save_pools"

    lock = threading.Lock()
    pools = {}  # name => pool
    pool_config = {}

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
    def set_pool_config(cls, config: dict):
        if not isinstance(config, dict):
            raise ValueError(f"config data must be dict but got {type(config)}")
        for k, v in config.items():
            cls.pool_config[k.lower()] = v

    @classmethod
    def _keep_hist_records(cls, name):
        name = name.lower()
        save_pools_list = cls.pool_config.get(cls._CONFIG_KEY_SAVE_POOLS, None)
        if not save_pools_list:
            return False

        return ("*" in save_pools_list) or (name in save_pools_list)

    @classmethod
    def add_time_hist_pool(cls, name: str, description: str, marks=None, scope=None):
        # check pool config
        keep_records = cls._keep_hist_records(name)
        name = cls._check_name(name, scope)
        p = new_time_pool(name, description, marks, keep_records=keep_records)
        cls.pools[name] = p
        return p

    @classmethod
    def add_msg_size_pool(cls, name: str, description: str, marks=None, scope=None):
        keep_records = cls._keep_hist_records(name)
        name = cls._check_name(name, scope)
        p = new_message_size_pool(name, description, marks, keep_records=keep_records)
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
    def get_records(cls):
        with cls.lock:
            result = {}
            for k in sorted(cls.pools.keys()):
                v = cls.pools[k]
                if isinstance(v, HistPool):
                    recs = v.get_records()
                    if recs:
                        result[k] = recs
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

    @classmethod
    def dump_summary(cls, file_name: str):
        stats_dict = cls.to_dict()
        json_string = json.dumps(stats_dict, indent=4)
        with open(file_name, "w") as f:
            f.write(json_string)

    @classmethod
    def dump_records(cls, csv_file_name: str):
        recs_dict = cls.get_records()
        if not recs_dict:
            return

        with open(csv_file_name, "w") as f:
            writer = csv.writer(f)

            # num_pools
            for pool_name, cat_recs in recs_dict.items():
                assert isinstance(cat_recs, dict)

                # pool_name, num_categories
                writer.writerow([cls._ROW_TYPE_POOL, pool_name])

                for cat_name, recs in cat_recs.items():
                    # cat_name, num_recs
                    assert isinstance(recs, list)
                    writer.writerow([cls._ROW_TYPE_CAT, cat_name])
                    for rec in recs:
                        assert isinstance(rec, list)
                        row = [cls._ROW_TYPE_REC]
                        row.extend(rec)
                        writer.writerow(row)

    @classmethod
    def read_records(cls, csv_file_name: str):
        pools = {}
        cats = None
        recs = None
        with open(csv_file_name) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    raise ValueError(f"'{csv_file_name}' is not a valid stats pool record file: row too short")
                row_type = row[0]
                if row_type == cls._ROW_TYPE_POOL:
                    pool_name = row[1]
                    cats = {}
                    pools[pool_name] = cats
                elif row_type == cls._ROW_TYPE_CAT:
                    if cats is None:
                        raise ValueError(f"'{csv_file_name}' is not a valid stats pool record file")
                    recs = []
                    cat_name = row[1]
                    cats[cat_name] = recs
                elif row_type == cls._ROW_TYPE_REC:
                    if recs is None:
                        raise ValueError(f"'{csv_file_name}' is not a valid stats pool record file")
                    rec = [float(r) for r in row[1:]]
                    recs.append(rec)
                else:
                    raise ValueError(
                        f"'{csv_file_name}' is not a valid stats pool record file: unknown row type {row_type}"
                    )
        return pools
