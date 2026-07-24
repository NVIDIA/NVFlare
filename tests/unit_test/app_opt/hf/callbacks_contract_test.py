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

from types import SimpleNamespace

from nvflare.app_opt.hf.callbacks import FLMetricsCallback


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, scalar, global_step=None):
        self.scalars.append((tag, scalar, global_step))


class _ItemScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def test_metrics_callback_streams_only_rank_zero_finite_scalars(monkeypatch):
    import nvflare.client.tracking as tracking

    writer = _Writer()
    monkeypatch.setattr(tracking, "SummaryWriter", lambda: writer, raising=False)
    task_state = SimpleNamespace(rank=0, metric_step=lambda step: step + 100)
    callback = FLMetricsCallback(task_state)

    callback.on_log(
        args=None,
        state=SimpleNamespace(global_step=3),
        control=None,
        logs={"loss": 0.5, "lr": _ItemScalar(0.01), "nan": float("nan"), "text": "ignore"},
    )

    assert writer.scalars == [("loss", 0.5, 103), ("lr", 0.01, 103)]


def test_metrics_callback_ignores_nonzero_ranks(monkeypatch):
    import nvflare.client.tracking as tracking

    def fail_writer():
        raise AssertionError("nonzero ranks must not create SummaryWriter")

    monkeypatch.setattr(tracking, "SummaryWriter", fail_writer, raising=False)
    task_state = SimpleNamespace(rank=1, metric_step=lambda step: step)
    callback = FLMetricsCallback(task_state)

    callback.on_log(
        args=None,
        state=SimpleNamespace(global_step=3),
        control=None,
        logs={"loss": 0.5},
    )
