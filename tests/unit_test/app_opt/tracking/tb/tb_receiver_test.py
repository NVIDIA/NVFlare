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

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver


def _make_fl_ctx(run_dir: Path) -> FLContext:
    workspace = Mock()
    workspace.get_run_dir.return_value = str(run_dir)

    engine = Mock()
    engine.get_workspace.return_value = workspace

    fl_ctx = FLContext()
    fl_ctx.get_engine = Mock(return_value=engine)
    fl_ctx.get_job_id = Mock(return_value="job-1")
    return fl_ctx


def _read_accumulator(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator


class TestTBAnalyticsReceiver:
    def test_save_scalar_preserves_step_zero(self, tmp_path):
        receiver = TBAnalyticsReceiver()
        fl_ctx = _make_fl_ctx(tmp_path / "run")
        receiver.initialize(fl_ctx)

        shareable = create_analytic_dxo(
            tag="loss",
            value=0.75,
            data_type=AnalyticsDataType.SCALAR,
            global_step=0,
        ).to_shareable()

        receiver.save(fl_ctx, shareable, "site-1")
        receiver.finalize(fl_ctx)

        accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1")
        assert accumulator.Tags()["scalars"] == ["loss"]
        events = accumulator.Scalars("loss")
        assert [(event.step, event.value) for event in events] == [(0, 0.75)]

    def test_save_text_writes_tensor_summary(self, tmp_path):
        receiver = TBAnalyticsReceiver()
        fl_ctx = _make_fl_ctx(tmp_path / "run")
        receiver.initialize(fl_ctx)

        shareable = create_analytic_dxo(
            tag="debug_msg",
            value="hello world",
            data_type=AnalyticsDataType.TEXT,
            global_step=2,
        ).to_shareable()

        receiver.save(fl_ctx, shareable, "site-1")
        receiver.finalize(fl_ctx)

        accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1")
        assert accumulator.Tags()["tensors"] == ["debug_msg"]
        event = accumulator.Tensors("debug_msg")[0]
        assert event.step == 2
        assert event.tensor_proto.string_val[0] == b"hello world"

    def test_save_image_writes_image_summary(self, tmp_path):
        receiver = TBAnalyticsReceiver()
        fl_ctx = _make_fl_ctx(tmp_path / "run")
        receiver.initialize(fl_ctx)

        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]

        shareable = create_analytic_dxo(
            tag="sample_image",
            value=image,
            data_type=AnalyticsDataType.IMAGE,
            global_step=1,
        ).to_shareable()

        receiver.save(fl_ctx, shareable, "site-1")
        receiver.finalize(fl_ctx)

        accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1")
        assert accumulator.Tags()["images"] == ["sample_image"]
        event = accumulator.Images("sample_image")[0]
        assert event.step == 1
        assert event.width == 2
        assert event.height == 2
        assert event.encoded_image_string

    def test_save_parameters_converts_scalars_and_text(self, tmp_path):
        receiver = TBAnalyticsReceiver()
        fl_ctx = _make_fl_ctx(tmp_path / "run")
        receiver.initialize(fl_ctx)

        shareable = create_analytic_dxo(
            tag="params",
            value={"lr": 0.1, "model": "cnn"},
            data_type=AnalyticsDataType.PARAMETERS,
        ).to_shareable()

        receiver.save(fl_ctx, shareable, "site-1")
        receiver.finalize(fl_ctx)

        accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1")
        assert accumulator.Scalars("lr")[0].value == pytest.approx(0.1)
        assert accumulator.Tensors("model")[0].tensor_proto.string_val[0] == b"cnn"

    def test_save_metric_alias_writes_scalar_summary(self, tmp_path):
        receiver = TBAnalyticsReceiver()
        fl_ctx = _make_fl_ctx(tmp_path / "run")
        receiver.initialize(fl_ctx)

        shareable = create_analytic_dxo(
            tag="accuracy",
            value=0.92,
            data_type=AnalyticsDataType.METRIC,
            global_step=3,
        ).to_shareable()

        receiver.save(fl_ctx, shareable, "site-1")
        receiver.finalize(fl_ctx)

        accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1")
        events = accumulator.Scalars("accuracy")
        assert len(events) == 1
        assert events[0].step == 3
        assert events[0].value == pytest.approx(0.92)

    def test_save_scalars_uses_per_series_subdirectories(self, tmp_path):
        receiver = TBAnalyticsReceiver()
        fl_ctx = _make_fl_ctx(tmp_path / "run")
        receiver.initialize(fl_ctx)

        shareable = create_analytic_dxo(
            tag="metrics",
            value={"train": 1.0, "val": 2.0},
            data_type=AnalyticsDataType.SCALARS,
            global_step=7,
        ).to_shareable()

        receiver.save(fl_ctx, shareable, "site-1")
        receiver.finalize(fl_ctx)

        train_accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1" / "metrics_train")
        val_accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1" / "metrics_val")

        assert [(event.step, event.value) for event in train_accumulator.Scalars("metrics")] == [(7, 1.0)]
        assert [(event.step, event.value) for event in val_accumulator.Scalars("metrics")] == [(7, 2.0)]

    def test_save_metrics_alias_uses_per_series_subdirectories(self, tmp_path):
        receiver = TBAnalyticsReceiver()
        fl_ctx = _make_fl_ctx(tmp_path / "run")
        receiver.initialize(fl_ctx)

        shareable = create_analytic_dxo(
            tag="eval_metrics",
            value={"precision": 0.8, "recall": 0.7},
            data_type=AnalyticsDataType.METRICS,
            global_step=5,
        ).to_shareable()

        receiver.save(fl_ctx, shareable, "site-1")
        receiver.finalize(fl_ctx)

        precision_accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1" / "eval_metrics_precision")
        recall_accumulator = _read_accumulator(tmp_path / "run" / "tb_events" / "site-1" / "eval_metrics_recall")

        precision_events = precision_accumulator.Scalars("eval_metrics")
        recall_events = recall_accumulator.Scalars("eval_metrics")

        assert len(precision_events) == 1
        assert precision_events[0].step == 5
        assert precision_events[0].value == pytest.approx(0.8)

        assert len(recall_events) == 1
        assert recall_events[0].step == 5
        assert recall_events[0].value == pytest.approx(0.7)
