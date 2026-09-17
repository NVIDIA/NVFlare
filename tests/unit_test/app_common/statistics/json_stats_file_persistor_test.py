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

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import StorageException
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.json_stats_file_persistor import OBJECT_ENCODER_PATH, JsonStatsFileWriter
from nvflare.app_common.workflows.statistics_controller import StatisticsController


class _FakeEngine:
    def __init__(self, components: dict):
        self.components = components

    def get_component(self, component_id: str):
        return self.components[component_id]


def _make_fl_ctx(tmp_path, byoc=False):
    app_root = tmp_path / "job" / "app"
    app_root.mkdir(parents=True)

    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.JOB_META, {AppValidationKey.BYOC: byoc}, private=True, sticky=False)
    return fl_ctx


def test_save_writes_relative_output_inside_job_dir(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    writer = JsonStatsFileWriter(output_path="statistics/result.json")

    writer.save({"count": {"site-1": 1}}, overwrite_existing=True, fl_ctx=fl_ctx)

    output_file = tmp_path / "job" / "statistics" / "result.json"
    assert json.loads(output_file.read_text(encoding="utf-8")) == {"count": {"site-1": 1}}


def test_save_accepts_existing_builtin_encoder_path(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    writer = JsonStatsFileWriter(output_path="statistics/result.json", json_encoder_path=OBJECT_ENCODER_PATH)

    writer.save({"count": {"site-1": 1}}, overwrite_existing=True, fl_ctx=fl_ctx)

    output_file = tmp_path / "job" / "statistics" / "result.json"
    assert json.loads(output_file.read_text(encoding="utf-8")) == {"count": {"site-1": 1}}


def test_rejects_non_string_json_encoder_path():
    with pytest.raises(TypeError, match="must be str"):
        JsonStatsFileWriter(output_path="result.json", json_encoder_path=123)


def test_save_rejects_absolute_output_path(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    outside_file = tmp_path / "outside.json"
    writer = JsonStatsFileWriter(output_path=str(outside_file))

    with pytest.raises(ValueError, match="must be relative"):
        writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)

    assert not outside_file.exists()


def test_save_rejects_parent_directory_traversal(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    outside_file = tmp_path / "outside.json"
    writer = JsonStatsFileWriter(output_path="../outside.json")

    with pytest.raises(ValueError, match="must stay inside"):
        writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)

    assert not outside_file.exists()


def test_save_rejects_symlink_escape(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    link_path = tmp_path / "job" / "link"
    try:
        link_path.symlink_to(outside_dir, target_is_directory=True)
    except OSError as ex:
        pytest.skip(f"symlink creation is not available: {ex}")

    writer = JsonStatsFileWriter(output_path="link/result.json")

    with pytest.raises(ValueError, match="must stay inside"):
        writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)

    assert not (outside_dir / "result.json").exists()


def test_save_rejects_missing_app_root():
    fl_ctx = FLContext()
    writer = JsonStatsFileWriter(output_path="result.json")

    with pytest.raises(ValueError, match=FLContextKey.APP_ROOT):
        writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)


def test_custom_encoder_path_requires_byoc(tmp_path, monkeypatch):
    fl_ctx = _make_fl_ctx(tmp_path, byoc=False)
    writer = JsonStatsFileWriter(output_path="statistics/result.json", json_encoder_path="json.JSONEncoder")

    def fail_load_class(_):
        raise AssertionError("custom encoder should not be loaded for non-BYOC jobs")

    monkeypatch.setattr("nvflare.app_common.statistics.json_stats_file_persistor.load_class", fail_load_class)

    with pytest.raises(ValueError, match="BYOC"):
        writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)

    assert not (tmp_path / "job" / "statistics" / "result.json").exists()


def test_custom_encoder_path_is_allowed_for_byoc(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path, byoc=True)
    writer = JsonStatsFileWriter(output_path="statistics/result.json", json_encoder_path="json.JSONEncoder")

    writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)

    output_file = tmp_path / "job" / "statistics" / "result.json"
    assert json.loads(output_file.read_text(encoding="utf-8")) == {"count": 1}


def test_custom_encoder_must_be_jsonencoder_subclass(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path, byoc=True)
    writer = JsonStatsFileWriter(output_path="result.json", json_encoder_path="subprocess.Popen")

    with pytest.raises(TypeError, match="JSONEncoder"):
        writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)

    assert not (tmp_path / "job" / "result.json").exists()


def test_save_respects_overwrite_existing_false(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    writer = JsonStatsFileWriter(output_path="statistics/result.json")
    writer.save({"count": 1}, overwrite_existing=True, fl_ctx=fl_ctx)

    with pytest.raises(StorageException, match="overwrite_existing is False"):
        writer.save({"count": 2}, overwrite_existing=False, fl_ctx=fl_ctx)

    output_file = tmp_path / "job" / "statistics" / "result.json"
    assert json.loads(output_file.read_text(encoding="utf-8")) == {"count": 1}


def test_save_does_not_create_parent_directory_when_json_serialization_fails(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    writer = JsonStatsFileWriter(output_path="statistics/result.json")

    with pytest.raises(TypeError):
        writer.save({"bad": object()}, overwrite_existing=True, fl_ctx=fl_ctx)

    assert not (tmp_path / "job" / "statistics").exists()


def test_statistics_controller_post_fn_writes_combined_fed_stats(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    writer = JsonStatsFileWriter(output_path="statistics/fed_stats.json", json_encoder_path=OBJECT_ENCODER_PATH)
    fl_ctx.set_prop(ReservedKey.ENGINE, _FakeEngine({"stats_writer": writer}), private=True, sticky=False)

    controller = StatisticsController(statistic_configs={StC.STATS_COUNT: {}}, writer_id="stats_writer", min_clients=2)
    controller.client_statistics = {
        StC.STATS_COUNT: {
            "site-1": {"train": {"Age": 10}},
            "site-2": {"train": {"Age": 20}},
        }
    }
    controller.global_statistics = {
        StC.STATS_COUNT: {
            "train": {"Age": 30},
        }
    }

    controller.post_fn(StC.FED_STATS_TASK, fl_ctx)

    output_file = tmp_path / "job" / "statistics" / "fed_stats.json"
    result = json.loads(output_file.read_text(encoding="utf-8"))
    assert result["Age"][StC.STATS_COUNT]["site-1"]["train"] == 10
    assert result["Age"][StC.STATS_COUNT]["site-2"]["train"] == 20
    assert result["Age"][StC.STATS_COUNT][StC.GLOBAL]["train"] == 30
