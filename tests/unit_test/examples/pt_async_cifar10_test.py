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

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

from nvflare.collab.api.exceptions import CollabCallError
from nvflare.collab.api.group_call_context import ResultQueue

pytest.importorskip("torch")
pytest.importorskip("tensorboard")
pytest.importorskip("torchvision")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_DIR = _REPO_ROOT / "examples" / "advanced" / "collab" / "pt_async_cifar10"


@pytest.fixture(scope="module")
def async_aggregator_module():
    original_model = sys.modules.pop("model", None)
    sys.path.insert(0, str(_EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "pt_async_cifar10_aggregator", _EXAMPLE_DIR / "async_aggregator.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.pop(0)
        if original_model is not None:
            sys.modules["model"] = original_model
        else:
            sys.modules.pop("model", None)


def test_group_failure_is_forwarded_before_other_outcomes(async_aggregator_module):
    aggregator = async_aggregator_module.Cifar10AsyncAggregator(data_root="/tmp")
    results = ResultQueue(limit=2, retain_history=False)
    watcher = threading.Thread(target=aggregator._watch_call_failures, args=(results,), daemon=True)
    watcher.start()

    error = CollabCallError("site-1", "train", RuntimeError("failed"))
    results.append_failure("site-1", error)

    outcome = aggregator._outcomes.get(timeout=1.0)
    assert outcome.physical_name == "site-1"
    assert outcome.error is error
    assert watcher.is_alive()

    results.append(("site-2", None))
    watcher.join(timeout=1.0)
    assert not watcher.is_alive()


def test_client_is_removed_after_consecutive_failure_limit(async_aggregator_module):
    aggregator = async_aggregator_module.Cifar10AsyncAggregator(data_root="/tmp", max_client_failures=2)
    job = async_aggregator_module._ActiveJob(
        physical_name="site-1",
        logical_name="site-1",
        assignment_id=0,
        model_version=0,
        base_model={},
    )

    aggregator._active_jobs["site-1"] = job
    aggregator._process_outcome(
        async_aggregator_module._ClientOutcome(physical_name="site-1", error=RuntimeError("first")),
        accept_update=True,
    )
    assert aggregator._available_clients == ["site-1"]

    aggregator._available_clients.clear()
    aggregator._active_jobs["site-1"] = job
    aggregator._process_outcome(
        async_aggregator_module._ClientOutcome(physical_name="site-1", error=RuntimeError("second")),
        accept_update=True,
    )
    assert aggregator._available_clients == []
    assert aggregator._client_failures["site-1"] == 2
