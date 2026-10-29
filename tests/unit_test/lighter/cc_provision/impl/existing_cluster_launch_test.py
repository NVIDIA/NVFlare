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


"""In-process CoCo policy and validation contracts."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tests.coco_test_utils.shell import ROOT


@pytest.mark.parametrize("failure", ["deadline", "request-timeout"])
def test_readiness_deadline_and_api_request_timeout_are_bounded(tmp_path, failure):
    applied = tmp_path / "applied.json"
    applied.write_text(json.dumps({"metadata": {"name": "review", "namespace": "default", "uid": "current-uid"}}))
    script = (ROOT / "coco/50-launch-handoff.sh").read_text()
    program = script.split("<<'PY'\n", 2)[2].split("\nPY\n", 1)[0]
    clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 601 if failure == "deadline" else 1]), sleep=Mock())
    request = Mock(side_effect=subprocess.TimeoutExpired("kubectl", 15))
    api = SimpleNamespace(run=request, TimeoutExpired=subprocess.TimeoutExpired)
    with (
        patch.dict(sys.modules, {"time": clock, "subprocess": api}),
        patch.object(sys, "argv", ["readiness", str(applied)]),
    ):
        with pytest.raises(SystemExit, match="timed out"):
            exec(compile(program, str(ROOT / "coco/50-launch-handoff.sh"), "exec"), {})
    if failure == "deadline":
        request.assert_not_called()
    else:
        assert request.call_args.kwargs["timeout"] <= 15
        assert "--request-timeout=10s" in request.call_args.args[0]
    clock.sleep.assert_not_called()
