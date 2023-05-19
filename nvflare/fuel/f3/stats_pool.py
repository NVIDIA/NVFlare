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
import sys
import threading
import time
from typing import List, Tuple, Union

_KEY_MAX = "max"
_KEY_MIN = "min"
_KEY_NAME = "name"
_KEY_DESC = "description"
_KEY_TOTAL = "total"
_KEY_COUNT = "count"
_KEY_UNIT = "unit"
_KEY_MARKS = "marks"
_KEY_COUNTER_NAMES = "counter_names"
_KEY_CAT_DATA = "cat_data"


class StatsMode:

    COUNT = "count"
    PERCENT = "percent"
    AVERAGE = "avg"
    MIN = "min"
    MAX = "max"


VALID_HIST_MODES = [StatsMode.COUNT, StatsMode.PERCENT, StatsMode.AVERAGE, StatsMode.MAX, StatsMode.MIN]


def format_value(v: float, n=3):
    if v is None:
        return "n/a"
    fmt = "{:." + str(n) + "e}"
    return fmt.format(v)


class _Bin:
    def __init__(self, count=0, total_value=0.0, min_value=None, max_value=None):
        self.count = count
        self.total = total_value
        self.min = min_value
        self.max = max_value

    def record_value(self, value: float):
        self.count += 1
        self.total += value
        if self.min is None or self.min > value:
            self.min = value
        if self.max is None or self.max < value:
            self.max = value

    def get_content(self, mode=StatsMode.COUNT, total_count=0):
        if self.count == 0:
            return ""
        if mode == StatsMode.COUNT:
            return str(self.count)
        if mode == StatsMode.PERCENT:
            return str(round(self.count / total_count, 2))
        if mode == StatsMode.AVERAGE:
            avg = self.total / self.count
            return format_value(avg)
        if mode == StatsMode.MIN:
            return format_value(self.min)
        if mode == StatsMode.MAX:
            return format_value(self.max)
        return "n/a"

    def to_dict(self) -> dict:
        return {
            _KEY_COUNT: self.count,
            _KEY_TOTAL: self.total,
            _KEY_MIN: self.min if self.min is not None else "",
            _KEY_MAX: self.max if self.max is not None else "",
        }

    @staticmethod
    def from_dict(d: dict):
        if not isinstance(d, dict):
            raise ValueError(f"d must be dict but got {type(d)}")
        b = _Bin()
        b.count = d.get(_KEY_COUNT, 0)
        b.total = d.get(_KEY_TOTAL, 0)
        m = d.get(_KEY_MIN)
        if isinstance(m, str):
            b.min = None
        else:
            b.min = m
        x = d.get(_KEY_MAX)
        if isinstance(x, str):
            b.max = None
        else:
            b.max = x
        return b


class StatsPool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def to_dict(self) -> dict:
        pass

    def get_table(self, mode):
        pass

    @staticmethod
    def from_dict(d: dict):
        pass


class RecordWriter:
    def write(self, pool_name: str, category: str, value: float, report_time: float):
        pass

    def close(self):
        pass


class HistPool(StatsPool):
    def __init__(self, name: str, description: str, marks: Union[List[float], Tuple], unit: str, record_writer=None):
        if record_writer:
            if not isinstance(record_writer, RecordWriter):
                raise TypeError(f"record_writer must be RecordWriter but got {type(record_writer)}")

        StatsPool.__init__(self, name, description)
        self.update_lock = threading.Lock()
        self.unit = unit
        self.marks = marks
        self.record_writer = record_writer  # used for writing raw records
        self.cat_bins = {}  # category name => list of bins

        if not marks:
            raise ValueError("marks not specified")
        if len(marks) < 2:
            raise ValueError(f"marks must have at least two numbers but got {len(marks)}")

        for i in range(1, len(marks)):
            if marks[i] <= marks[i - 1]:
                raise ValueError(f"marks must contain increasing values, but got {marks}")

        # A range is defined: left <= N < right  [...)
        # [..., M1) [M1, M2) [M2, M3) [M3, ...)
        m = sys.float_info.max
        self.ranges = [(-m, marks[0])]
        self.range_names = [f"<{marks[0]}"]
        for i in range(len(marks) - 1):
            self.ranges.append((marks[i], marks[i + 1]))
            self.range_names.append(f"{marks[i]}-{marks[i+1]}")
        self.ranges.append((marks[-1], m))
        self.range_names.append(f">={marks[-1]}")

    def record_value(self, category: str, value: float):
        with self.update_lock:
            bins = self.cat_bins.get(category)
            if bins is None:
                bins = [None for _ in range(len(self.ranges))]
                self.cat_bins[category] = bins

            for i in range(len(self.ranges)):
                r = self.ranges[i]
                if r[0] <= value < r[1]:
                    b = bins[i]
                    if not b:
                        b = _Bin()
                        bins[i] = b
                    b.record_value(value)

            if self.record_writer:
                self.record_writer.write(pool_name=self.name, category=category, value=value, report_time=time.time())

    def get_table(self, mode=StatsMode.COUNT):
        with self.update_lock:
            headers = ["category"]
            has_values = [False for _ in range(len(self.ranges))]

            # determine bins that have values in any category
            for _, bins in self.cat_bins.items():
                for i in range(len(self.ranges)):
                    if bins[i]:
                        has_values[i] = True

            for i in range(len(self.ranges)):
                if has_values[i]:
                    headers.append(self.range_names[i])

            headers.append("overall")

            rows = []
            for cat_name in sorted(self.cat_bins.keys()):
                bins = self.cat_bins[cat_name]
                total_count = 0
                total_value = 0.0
                overall_min = None
                overall_max = None

                for b in bins:
                    if b:
                        total_count += b.count
                        total_value += b.total
                        if b.max is not None:
                            if overall_max is None or overall_max < b.max:
                                overall_max = b.max

                        if b.min is not None:
                            if overall_min is None or overall_min > b.min:
                                overall_min = b.min

                r = [cat_name]
                for i in range(len(bins)):
                    if not has_values[i]:
                        continue

                    b = bins[i]
                    if not b:
                        r.append("")
                    else:
                        r.append(b.get_content(mode, total_count))

                # compute overall values
                overall_bin = _Bin(
                    count=total_count, total_value=total_value, max_value=overall_max, min_value=overall_min
                )
                r.append(overall_bin.get_content(mode, total_count))

                rows.append(r)
            return headers, rows

    def to_dict(self):
        with self.update_lock:
            cat_bins = {}
            for cat, bins in self.cat_bins.items():
                exp_bins = []
                for b in bins:
                    if not b:
                        exp_bins.append("")
                    else:
                        exp_bins.append(b.to_dict())
                cat_bins[cat] = exp_bins
            return {
                _KEY_NAME: self.name,
                _KEY_DESC: self.description,
                _KEY_MARKS: list(self.marks),
                _KEY_UNIT: self.unit,
                _KEY_CAT_DATA: cat_bins,
            }

    @staticmethod
    def from_dict(d: dict):
        p = HistPool(
            name=d.get(_KEY_NAME, ""),
            description=d.get(_KEY_DESC, ""),
            unit=d.get(_KEY_UNIT, ""),
            marks=d.get(_KEY_MARKS),
        )
        cat_bins = d.get(_KEY_CAT_DATA)
        if not cat_bins:
            return p

        for cat, bins in cat_bins.items():
            in_bins = []
            for b in bins:
                if not b:
                    in_bins.append(None)
                else:
                    assert isinstance(b, dict)
                    in_bins.append(_Bin.from_dict(b))
            p.cat_bins[cat] = in_bins
        return p


class CounterPool(StatsPool):
    def __init__(self, name: str, description: str, counter_names: List[str], dynamic_counter_name=True):
        if not counter_names and not dynamic_counter_name:
            raise ValueError("counter_names cannot be empty")
        StatsPool.__init__(self, name, description)
        self.counter_names = counter_names
        self.cat_counters = {}  # dict of cat_name => counter dict (counter_name => int)
        self.dynamic_counter_name = dynamic_counter_name
        self.update_lock = threading.Lock()

    def increment(self, category: str, counter_name: str, amount=1):
        with self.update_lock:
            if counter_name not in self.counter_names:
                if self.dynamic_counter_name:
                    self.counter_names.append(counter_name)
                else:
                    raise ValueError(f"'{counter_name}' is not defined in pool '{self.name}'")

            counters = self.cat_counters.get(category)
            if not counters:
                counters = {}
                self.cat_counters[category] = counters
            c = counters.get(counter_name, 0)
            c += amount
            counters[counter_name] = c

    def get_table(self, mode=""):
        with self.update_lock:
            headers = ["category"]
            eff_counter_names = []
            for cn in self.counter_names:
                for _, counters in self.cat_counters.items():
                    v = counters.get(cn, 0)
                    if v > 0:
                        eff_counter_names.append(cn)
                        break

            headers.extend(eff_counter_names)
            rows = []
            for cat_name in sorted(self.cat_counters.keys()):
                counters = self.cat_counters[cat_name]
                r = [cat_name]
                for cn in eff_counter_names:
                    value = counters.get(cn, 0)
                    r.append(str(value))
                rows.append(r)
            return headers, rows

    def to_dict(self):
        with self.update_lock:
            return {
                _KEY_NAME: self.name,
                _KEY_DESC: self.description,
                _KEY_COUNTER_NAMES: list(self.counter_names),
                _KEY_CAT_DATA: self.cat_counters,
            }

    @staticmethod
    def from_dict(d: dict):
        p = CounterPool(
            name=d.get(_KEY_NAME, ""), description=d.get(_KEY_DESC, ""), counter_names=d.get(_KEY_COUNTER_NAMES)
        )
        p.cat_counters = d.get(_KEY_CAT_DATA)
        return p


def new_time_pool(name: str, description="", marks=None, record_writer=None) -> HistPool:
    if not marks:
        marks = (0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0)
    return HistPool(name=name, description=description, marks=marks, unit="second", record_writer=record_writer)


def new_message_size_pool(name: str, description="", marks=None, record_writer=None) -> HistPool:
    if not marks:
        marks = (0.01, 0.1, 1, 10, 50, 100, 200, 500, 800, 1000)
    return HistPool(name=name, description=description, marks=marks, unit="MB", record_writer=record_writer)


def parse_hist_mode(mode: str) -> str:
    if not mode:
        return StatsMode.COUNT

    if mode.startswith("p"):
        return StatsMode.PERCENT
    elif mode.startswith("c"):
        return StatsMode.COUNT
    elif mode.startswith("a"):
        return StatsMode.AVERAGE

    if mode not in VALID_HIST_MODES:
        return ""
    else:
        return mode


class StatsPoolManager:

    _CONFIG_KEY_SAVE_POOLS = "save_pools"

    lock = threading.Lock()
    pools = {}  # name => pool
    pool_config = {}
    record_writer = None

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
    def set_record_writer(cls, record_writer: RecordWriter):
        if not isinstance(record_writer, RecordWriter):
            raise TypeError(f"record_writer must be RecordWriter but got {type(record_writer)}")
        cls.record_writer = record_writer

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
        record_writer = cls.record_writer if keep_records else None
        p = new_time_pool(name, description, marks, record_writer=record_writer)
        cls.pools[name] = p
        return p

    @classmethod
    def add_msg_size_pool(cls, name: str, description: str, marks=None, scope=None):
        keep_records = cls._keep_hist_records(name)
        name = cls._check_name(name, scope)
        record_writer = cls.record_writer if keep_records else None
        p = new_message_size_pool(name, description, marks, record_writer=record_writer)
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

    @classmethod
    def dump_summary(cls, file_name: str):
        stats_dict = cls.to_dict()
        json_string = json.dumps(stats_dict, indent=4)
        with open(file_name, "w") as f:
            f.write(json_string)

    @classmethod
    def close(cls):
        if cls.record_writer:
            cls.record_writer.close()


class CsvRecordHandler(RecordWriter):
    def __init__(self, file_name):
        self.file = open(file_name, "w")
        self.writer = csv.writer(self.file)
        self.lock = threading.Lock()

    def write(self, pool_name: str, category: str, value: float, report_time: float):
        if not pool_name.isascii():
            raise ValueError(f"pool_name {pool_name} contains non-ascii chars")
        if not category.isascii():
            raise ValueError(f"category {category} contains non-ascii chars")

        row = [pool_name, category, report_time, value]
        with self.lock:
            self.writer.writerow(row)

    def close(self):
        self.file.close()

    @staticmethod
    def read_records(csv_file_name: str):
        pools = {}
        reader = CsvRecordReader(csv_file_name)
        for rec in reader:
            pool_name = rec.pool_name
            cat_name = rec.category
            report_time = rec.report_time
            value = rec.value

            cats = pools.get(pool_name)
            if not cats:
                cats = {}
                pools[pool_name] = cats
            recs = cats.get(cat_name)
            if not recs:
                recs = []
                cats[cat_name] = recs
            recs.append((report_time, value))
        return pools


class StatsRecord:
    def __init__(self, pool_name, category, report_time, value):
        self.pool_name = pool_name
        self.category = category
        self.report_time = report_time
        self.value = value


class CsvRecordReader:
    def __init__(self, csv_file_name: str):
        self.csv_file_name = csv_file_name
        self.file = open(csv_file_name)
        self.reader = csv.reader(self.file)

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self.reader)
        if len(row) != 4:
            raise ValueError(f"'{self.csv_file_name}' is not a valid stats pool record file: bad row length {len(row)}")
        pool_name = row[0]
        cat_name = row[1]
        report_time = float(row[2])
        value = float(row[3])
        return StatsRecord(pool_name, cat_name, report_time, value)
