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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval


def _make_engine_and_ctx(run_dir):
    workspace = Mock()
    workspace.get_run_dir.return_value = str(run_dir)
    engine = Mock()
    engine.get_workspace.return_value = workspace

    fl_ctx = FLContext()
    fl_ctx.get_job_id = Mock(return_value="job-1")
    return engine, fl_ctx


class TestCrossSiteModelEvalPaths:
    @pytest.mark.parametrize("cross_val_dir", ["/tmp/outside_cse", "../outside_cse"])
    def test_start_controller_rejects_escaping_cross_val_dir(self, tmp_path, cross_val_dir):
        engine, fl_ctx = _make_engine_and_ctx(tmp_path / "run")
        controller = CrossSiteModelEval(cross_val_dir=cross_val_dir, participating_clients=["site-1"])
        controller._engine = engine

        with pytest.raises(ValueError, match="must (be relative|stay inside)"):
            controller.start_controller(fl_ctx)

    def test_start_controller_accepts_relative_cross_val_dir(self, tmp_path):
        engine, fl_ctx = _make_engine_and_ctx(tmp_path / "run")
        controller = CrossSiteModelEval(participating_clients=["site-1"])
        controller._engine = engine
        controller.fire_event = Mock()  # event plumbing is not under test

        controller.start_controller(fl_ctx)

        run_dir = os.path.realpath(str(tmp_path / "run"))
        assert os.path.isdir(os.path.join(run_dir, AppConstants.CROSS_VAL_DIR, AppConstants.CROSS_VAL_MODEL_DIR_NAME))
        assert os.path.isdir(os.path.join(run_dir, AppConstants.CROSS_VAL_DIR, AppConstants.CROSS_VAL_RESULTS_DIR_NAME))
