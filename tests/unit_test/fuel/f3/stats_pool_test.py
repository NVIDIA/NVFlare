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

import json

import pytest

from nvflare.fuel.f3.stats_pool import (
    CounterPool,
    CsvRecordHandler,
    CsvRecordReader,
    HistPool,
    RecordWriter,
    StatsMode,
    StatsPool,
    StatsPoolManager,
    _Bin,
    format_value,
    new_message_size_pool,
    new_time_pool,
    parse_hist_mode,
)


class _Writer(RecordWriter):
    def __init__(self):
        self.records = []
        self.closed = False

    def write(self, pool_name, category, value, report_time):
        self.records.append((pool_name, category, value, report_time))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_manager():
    StatsPoolManager.pools = {}
    StatsPoolManager.pool_config = {}
    StatsPoolManager.record_writer = None
    yield
    StatsPoolManager.pools = {}
    StatsPoolManager.pool_config = {}
    StatsPoolManager.record_writer = None


def test_bin_records_formats_and_round_trips():
    bin_ = _Bin()
    assert bin_.get_content() == ""

    for value in (3.0, 1.0, 5.0):
        bin_.record_value(value)

    assert bin_.get_content(StatsMode.COUNT) == "3"
    assert bin_.get_content(StatsMode.PERCENT, 6) == "0.5"
    assert bin_.get_content(StatsMode.AVERAGE) == format_value(3.0)
    assert bin_.get_content(StatsMode.MIN) == format_value(1.0)
    assert bin_.get_content(StatsMode.MAX) == format_value(5.0)
    assert bin_.get_content("unknown") == "n/a"
    restored = _Bin.from_dict(bin_.to_dict())
    assert (restored.count, restored.total, restored.min, restored.max) == (3, 9.0, 1.0, 5.0)

    with pytest.raises(ValueError, match="d must be dict"):
        _Bin.from_dict([])
    assert _Bin.from_dict({"min": "", "max": ""}).min is None
    assert format_value(None) == "n/a"


@pytest.mark.parametrize(
    "marks, error",
    [([], "marks not specified"), ([1], "at least two"), ([1, 1], "increasing values"), ([2, 1], "increasing values")],
)
def test_hist_pool_validates_marks(marks, error):
    with pytest.raises(ValueError, match=error):
        HistPool("pool", "description", marks, "unit")


def test_hist_pool_records_tables_and_round_trips():
    writer = _Writer()
    pool = HistPool("latency", "request latency", [1.0, 2.0], "second", writer)
    for category, value in (("b", 0.5), ("a", 1.0), ("a", 1.5), ("a", 3.0)):
        pool.record_value(category, value)

    headers, count_rows = pool.get_table(StatsMode.COUNT)
    assert headers == ["category", "<1.0", "1.0-2.0", ">=2.0", "overall"]
    assert count_rows == [["a", "", "2", "1", "3"], ["b", "1", "", "", "1"]]
    assert pool.get_table(StatsMode.MIN)[1][0][-1] == format_value(1.0)
    assert pool.get_table(StatsMode.MAX)[1][0][-1] == format_value(3.0)
    assert pool.get_table(StatsMode.AVERAGE)[1][0][-1] == format_value(5.5 / 3)
    assert pool.get_table(StatsMode.PERCENT)[1][0][2:] == ["0.67", "0.33", "1.0"]
    assert len(writer.records) == 4

    restored = HistPool.from_dict(pool.to_dict())
    assert restored.to_dict() == pool.to_dict()
    assert HistPool.from_dict({"name": "empty", "description": "empty", "marks": [1, 2], "unit": "s"}).cat_bins == {}


def test_hist_pool_rejects_invalid_writer_and_factories_supply_defaults():
    with pytest.raises(TypeError, match="record_writer must be RecordWriter"):
        HistPool("pool", "description", [1, 2], "unit", object())

    assert new_time_pool("time").unit == "second"
    assert new_message_size_pool("size").unit == "MB"
    assert new_time_pool("custom", marks=[1, 2]).marks == [1, 2]


def test_counter_pool_dynamic_and_fixed_counters_round_trip():
    pool = CounterPool("messages", "message results", ["ok"], dynamic_counter_name=True)
    pool.increment("site-b", "ok", 2)
    pool.increment("site-a", "error")
    pool.increment("site-a", "ok", 0)

    assert pool.counter_names == ["ok", "error"]
    assert pool.get_table() == (["category", "ok", "error"], [["site-a", "0", "1"], ["site-b", "2", "0"]])
    restored = CounterPool.from_dict(pool.to_dict())
    assert restored.to_dict() == pool.to_dict()

    fixed = CounterPool("fixed", "fixed", ["ok"], dynamic_counter_name=False)
    with pytest.raises(ValueError, match="not defined"):
        fixed.increment("site", "error")
    with pytest.raises(ValueError, match="cannot be empty"):
        CounterPool("bad", "bad", [], dynamic_counter_name=False)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", StatsMode.COUNT),
        ("percent", StatsMode.PERCENT),
        ("count", StatsMode.COUNT),
        ("avg", StatsMode.AVERAGE),
        ("min", StatsMode.MIN),
        ("max", StatsMode.MAX),
        ("bad", ""),
    ],
)
def test_parse_hist_mode(value, expected):
    assert parse_hist_mode(value) == expected


def test_manager_creates_scoped_pools_and_serializes(tmp_path):
    writer = _Writer()
    StatsPoolManager.set_record_writer(writer)
    StatsPoolManager.set_pool_config({"save_pools": ["*"]})

    hist = StatsPoolManager.add_time_hist_pool("Latency", "latency", marks=[1, 2])
    scoped = StatsPoolManager.add_time_hist_pool("Latency", "scoped", marks=[1, 2], scope="site")
    size = StatsPoolManager.add_msg_size_pool("Sizes", "sizes", marks=[1, 2])
    counter = StatsPoolManager.add_counter_pool("Results", "results", ["ok"])
    hist.record_value("site", 1.5)
    size.record_value("site", 0.5)
    counter.increment("site", "ok")

    assert scoped.name == "latency@site"
    assert hist.record_writer is writer
    assert StatsPoolManager.get_pool("LATENCY") is hist
    headers, rows = StatsPoolManager.get_table()
    assert headers == ["pool", "type", "description"]
    assert {row[1] for row in rows} == {"hist", "counter"}

    serialized = StatsPoolManager.to_dict()
    StatsPoolManager.from_dict(serialized)
    assert set(StatsPoolManager.pools) == {"latency", "latency@site", "sizes", "results"}

    output = tmp_path / "stats.json"
    StatsPoolManager.dump_summary(str(output))
    assert json.loads(output.read_text()) == StatsPoolManager.to_dict()
    assert StatsPoolManager.delete_pool("RESULTS").name == "results"
    assert StatsPoolManager.delete_pool("missing") is None
    StatsPoolManager.close()
    assert writer.closed


def test_manager_validates_configuration_and_serialized_data():
    with pytest.raises(ValueError, match="config data must be dict"):
        StatsPoolManager.set_pool_config([])
    with pytest.raises(TypeError, match="record_writer must be RecordWriter"):
        StatsPoolManager.set_record_writer(object())

    StatsPoolManager.add_counter_pool("pool", "description", ["ok"])
    with pytest.raises(ValueError, match="already defined"):
        StatsPoolManager.add_counter_pool("POOL", "description", ["ok"])
    with pytest.raises(ValueError, match="missing pool type"):
        StatsPoolManager.from_dict({"pool": {"pool": {}}})
    with pytest.raises(ValueError, match="missing pool data"):
        StatsPoolManager.from_dict({"pool": {"type": "hist"}})
    with pytest.raises(ValueError, match="invalid pool type"):
        StatsPoolManager.from_dict({"pool": {"type": "other", "pool": {"value": 1}}})


def test_csv_record_handler_writes_reads_and_validates(tmp_path):
    path = tmp_path / "records.csv"
    handler = CsvRecordHandler(str(path))
    handler.write("latency", "site-1", 1.5, 10.0)
    handler.write("latency", "site-1", 2.5, 11.0)
    handler.close()

    records = list(CsvRecordReader(str(path)))
    assert [(r.pool_name, r.category, r.report_time, r.value) for r in records] == [
        ("latency", "site-1", 10.0, 1.5),
        ("latency", "site-1", 11.0, 2.5),
    ]
    assert CsvRecordHandler.read_records(str(path)) == {"latency": {"site-1": [(10.0, 1.5), (11.0, 2.5)]}}

    invalid = CsvRecordHandler(str(tmp_path / "invalid.csv"))
    with pytest.raises(ValueError, match="non-ascii"):
        invalid.write("latencÿ", "site", 1.0, 1.0)
    with pytest.raises(ValueError, match="non-ascii"):
        invalid.write("latency", "sité", 1.0, 1.0)
    invalid.close()

    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("only,two\n")
    with pytest.raises(ValueError, match="bad row length"):
        next(CsvRecordReader(str(bad_path)))


def test_base_interfaces_are_no_ops():
    pool = StatsPool("pool", "description")
    writer = RecordWriter()
    assert pool.to_dict() is None
    assert pool.get_table(StatsMode.COUNT) is None
    assert pool.from_dict({}) is None
    assert writer.write("pool", "category", 1.0, 1.0) is None
    assert writer.close() is None
