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
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator


def _make_fl_ctx(run_dir):
    workspace = Mock()
    workspace.get_run_dir.return_value = str(run_dir)
    engine = Mock()
    engine.get_workspace.return_value = workspace

    fl_ctx = FLContext()
    fl_ctx.get_engine = Mock(return_value=engine)
    fl_ctx.get_job_id = Mock(return_value="job-1")
    return fl_ctx


class TestValidationJsonGeneratorPaths:
    def test_end_run_writes_results_under_run_dir(self, tmp_path):
        generator = ValidationJsonGenerator()
        fl_ctx = _make_fl_ctx(tmp_path / "run")

        generator.handle_event(EventType.END_RUN, fl_ctx)

        out = os.path.join(
            os.path.realpath(str(tmp_path / "run")), AppConstants.CROSS_VAL_DIR, "cross_val_results.json"
        )
        assert os.path.isfile(out)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"results_dir": "/tmp/outside_val"},
            {"results_dir": "../outside_val"},
            {"json_file_name": "/tmp/outside.json"},
        ],
    )
    def test_end_run_rejects_escaping_paths(self, tmp_path, kwargs):
        generator = ValidationJsonGenerator(**kwargs)
        fl_ctx = _make_fl_ctx(tmp_path / "run")

        with pytest.raises(ValueError, match="must (be relative|stay inside)"):
            generator.handle_event(EventType.END_RUN, fl_ctx)
