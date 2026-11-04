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

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.utils.analytix_utils import create_analytic_dxo, send_analytic_dxo
from nvflare.apis.utils.fl_context_utils import add_job_audit_event


class MemorizationAuditReporter(FLComponent):
    """Persist local aggregate audit reports and optionally emit selected scalars.

    The same reporter can be used by periodic filters and final evaluation jobs.
    Detailed record scores and raw text must be stored separately by the auditor.
    """

    def __init__(self, report_dir="memorization_audits", summary_metrics=None):
        super().__init__()
        if Path(report_dir).is_absolute() or ".." in Path(report_dir).parts:
            raise ValueError("report_dir must be relative to the job workspace")
        if summary_metrics is not None and (
            not isinstance(summary_metrics, list) or any(not isinstance(name, str) for name in summary_metrics)
        ):
            raise ValueError("summary_metrics must be a list of metric names")
        self.report_dir = report_dir
        self.summary_metrics = summary_metrics or []

    def report(self, report, fl_ctx, stage="client_task_result"):
        report = dict(report)
        report.update(
            schema_version=1,
            job_id=fl_ctx.get_job_id(),
            task_id=fl_ctx.get_prop(FLContextKey.TASK_ID),
            site=fl_ctx.get_identity_name(),
            stage=stage,
        )
        try:
            workspace = fl_ctx.get_engine().get_workspace()
            directory = Path(workspace.get_run_dir(fl_ctx.get_job_id())) / self.report_dir
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            identity = [
                report["job_id"],
                report["task_id"],
                stage,
                report.get("round"),
                report.get("model_fingerprint"),
            ]
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            path = directory / f"{key}.json"
            data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as stream:
                    stream.write(data)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            add_job_audit_event(fl_ctx, msg=f"memorization audit {report['status']}: {path.name}")
            if report["status"] == "completed":
                for name in self.summary_metrics:
                    value = report.get("metrics", {}).get(name)
                    if isinstance(value, (int, float)):
                        dxo = create_analytic_dxo(
                            f"memorization/{name}", value, AnalyticsDataType.SCALAR, global_step=report["round"]
                        )
                        send_analytic_dxo(self, dxo, fl_ctx)
            self.log_info(fl_ctx, f"Memorization audit {report['status']}: {path}")
            return path
        except Exception as exc:
            self.log_error(fl_ctx, f"Memorization audit reporting failed: {type(exc).__name__}")
            return None
