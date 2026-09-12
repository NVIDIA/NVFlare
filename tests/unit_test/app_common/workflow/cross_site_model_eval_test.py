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


@pytest.mark.parametrize("data_kind", ["METRICS", "WEIGHTS"])
def test_controller_only_evaluation_reports_bounded_results(tmp_path, caplog, data_kind):
    import logging

    from nvflare.apis.dxo import DXO, DataKind, from_file

    engine, ctx = _make_engine_and_ctx(tmp_path)
    controller = CrossSiteModelEval(participating_clients=["site-1"])
    controller._engine = engine
    controller.fire_event = Mock()
    controller.start_controller(ctx)
    payload = {"accuracy": 0.75, "samples": list(range(100000)), "detail": "X" * 100000}
    payload.update({f"extra_{n}": n for n in range(100)})
    with caplog.at_level(logging.INFO):
        controller._save_validation_result("site-1", "global.pt", DXO(getattr(DataKind, data_kind), payload), ctx)
    progress = [r.message for r in caplog.records if r.name.endswith(".progress")]
    assert len(progress) == 1
    assert 'Evaluated "global.pt" on "site-1"' in progress[0]
    assert len(progress[0]) < 1024
    assert "see saved result" in progress[0]
    if data_kind == "METRICS":
        assert "accuracy=0.75" in progress[0]
    else:
        assert "accuracy=" not in progress[0]
    assert not list(tmp_path.rglob("cross_val_results.json"))
    saved = from_file(controller._val_results["site-1"]["global.pt"])
    assert len(saved.data["samples"]) == 100000
    assert len(saved.data["detail"]) == 100000
