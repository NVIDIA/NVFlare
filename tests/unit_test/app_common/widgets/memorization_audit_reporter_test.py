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
from unittest.mock import Mock

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.widgets.memorization_audit_reporter import MemorizationAuditReporter


def test_local_reports_permissions_retry_and_explicit_summary_selection(tmp_path, monkeypatch):
    fl_ctx = FLContext()
    engine = Mock()
    engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    fl_ctx.set_prop(ReservedKey.ENGINE, engine)
    fl_ctx.set_prop(FLContextKey.CURRENT_RUN, "test-job")
    fl_ctx.set_prop(FLContextKey.TASK_ID, "task")
    send = Mock()
    monkeypatch.setattr("nvflare.app_common.widgets.memorization_audit_reporter.send_analytic_dxo", send)
    reporter = MemorizationAuditReporter(summary_metrics=["approved"])
    result = {"status": "completed", "round": 0, "model_fingerprint": "sha", "metrics": {"approved": 0.5, "local": 0.8}}
    first = reporter.report(result, fl_ctx)
    assert first.stat().st_mode & 0o777 == 0o600
    assert "job_id" not in result
    assert send.call_count == 1
    assert reporter.report(result, fl_ctx) == first
    final = reporter.report(result, fl_ctx, stage="global_model")
    assert final != first
    assert json.loads(final.read_text())["stage"] == "global_model"
