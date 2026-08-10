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

import os
import stat
from unittest.mock import patch

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.dxo import from_dict
from nvflare.app_common.metrics_exchange.metrics_sender import (
    ANALYTICS_BOOTSTRAP_ENV,
    CHANNEL,
    TOPIC_LOG,
    MetricsSender,
    read_bootstrap,
    resolve_bootstrap,
    write_bootstrap,
)
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply


def _config():
    return {
        "connect_url": "shared-file://0/tmp/local",
        "receiver_fqcn": "site-1.job",
        "client_fqcn": "site-1.job.metrics~abc",
    }


def test_bootstrap_is_atomic_owner_only_and_detects_env_conflicts(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text("old")
    write_bootstrap(str(path), _config())

    assert read_bootstrap(str(path))[1] == _config()
    assert list(tmp_path.glob(".analytics-*.tmp")) == []
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    monkeypatch.setenv(ANALYTICS_BOOTSTRAP_ENV, str(tmp_path / "other.json"))
    with pytest.raises(ValueError, match="conflict"):
        resolve_bootstrap(str(path))


@pytest.mark.parametrize("field,value", [("connect_url", ""), ("client_fqcn", "other.client")])
def test_invalid_bootstrap_is_rejected(tmp_path, field, value):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError):
        write_bootstrap(str(tmp_path / "bad.json"), config)


def test_sender_rejects_ambiguous_config(tmp_path):
    with pytest.raises(ValueError, match="either config or config_file"):
        MetricsSender(config=_config(), config_file=str(tmp_path / "metrics.json"))


class _Cell:
    def __init__(self):
        self.sent = []
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def is_cell_connected(self, target):
        return True

    def send_request(self, channel, topic, target, request, timeout):
        self.sent.append((channel, topic, target, request, timeout))
        return make_reply(CellReturnCode.OK)


def test_sender_uses_a_normal_credential_free_cell_and_sends_dxo(tmp_path):
    path = tmp_path / "metrics.json"
    write_bootstrap(str(path), _config())
    cell = _Cell()
    with patch("nvflare.app_common.metrics_exchange.metrics_sender.Cell", return_value=cell) as cell_cls:
        sender = MetricsSender(config_file=str(path))
        sender.init("0")
        assert sender.add("loss", 0.25, AnalyticsDataType.SCALAR, global_step=3)
        sender.shutdown()

    assert cell_cls.call_args.kwargs == {
        "fqcn": _config()["client_fqcn"],
        "root_url": None,
        "secure": False,
        "credentials": {},
        "parent_url": _config()["connect_url"],
    }
    channel, topic, target, request, _ = cell.sent[0]
    assert (channel, topic, target) == (CHANNEL, TOPIC_LOG, _config()["receiver_fqcn"])
    assert request.headers == {}
    dxo = from_dict(request.payload)
    assert (dxo.data["track_key"], dxo.data["track_value"], dxo.data["global_step"]) == ("loss", 0.25, 3)
    assert cell.stopped


def test_nonzero_rank_does_not_create_a_cell():
    with patch("nvflare.app_common.metrics_exchange.metrics_sender.Cell") as cell_cls:
        sender = MetricsSender(config=_config())
        sender.init("1")
        assert not sender.add("loss", 0.25, AnalyticsDataType.SCALAR)
    cell_cls.assert_not_called()
