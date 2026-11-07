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
import time
from copy import deepcopy
from pathlib import Path

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_common.abstract.memorization_auditor import MemorizationAuditor, UnsupportedAudit
from nvflare.app_common.filters.model_filter import ModelFilter
from nvflare.app_common.widgets.memorization_audit_reporter import MemorizationAuditReporter
from nvflare.fuel.utils.experimental import experimental


def _task_key(fl_ctx, round_number):
    return (fl_ctx.get_job_id(), fl_ctx.get_prop(FLContextKey.TASK_ID), round_number)


class AuditModelSnapshot(FLComponent):
    """Private baseline store for sequential client training tasks; never transmitted."""

    @experimental("AuditModelSnapshot API may change")
    def __init__(self):
        super().__init__()
        self._key = None
        self._model = None

    def capture(self, model, fl_ctx):
        if model.params_type != ParamsType.FULL or model.current_round is None:
            raise UnsupportedAudit("Baseline requires full parameters and a round identity")
        self._key = _task_key(fl_ctx, model.current_round)
        self._model = deepcopy(model)

    def pop(self, model, fl_ctx):
        result = self._model if self._key == _task_key(fl_ctx, model.current_round) else None
        self.clear()
        return result

    def clear(self):
        self._key, self._model = None, None

    def handle_event(self, event_type, fl_ctx):
        if event_type in (EventType.START_RUN, EventType.END_RUN):
            self.clear()


class AuditSnapshotFilter(ModelFilter):
    """Capture the incoming baseline after input transformations for a paired audit."""

    @experimental("AuditSnapshotFilter API may change")
    def __init__(self, snapshot_id: str):
        super().__init__([DataKind.WEIGHTS, DataKind.FL_MODEL])
        self.snapshot_id = snapshot_id

    def process_model(self, model, fl_ctx):
        snapshot = fl_ctx.get_engine().get_component(self.snapshot_id)
        if not isinstance(snapshot, AuditModelSnapshot):
            raise TypeError("snapshot_id must name an AuditModelSnapshot")
        snapshot.capture(model, fl_ctx)
        return model

    def process(self, shareable, fl_ctx):
        try:
            if from_shareable(shareable).data_kind == DataKind.COLLECTION:
                raise UnsupportedAudit("Audit snapshot requires a single model")
            return super().process(shareable, fl_ctx)
        except Exception as exc:
            snapshot = fl_ctx.get_engine().get_component(self.snapshot_id)
            if isinstance(snapshot, AuditModelSnapshot):
                snapshot.clear()
            self.log_warning(fl_ctx, f"Audit baseline capture failed: {type(exc).__name__}")
            return shareable


class MemorizationAuditFilter(ModelFilter):
    """Experimental, report-only outgoing model audit. Never suppresses an update.

    Register an auditor and optional snapshot store as job components. Reports are
    local JSON under the job workspace. Only explicitly selected scalar metrics
    are emitted to analytics; detailed audit data belongs in the auditor's local
    storage. Runtime failures log a failed report and leave training unaffected.
    """

    @experimental("MemorizationAuditFilter API may change")
    def __init__(
        self,
        auditor_id: str,
        snapshot_id: str | None = None,
        start_round: int = 0,
        frequency: int = 1,
        report_dir: str = "memorization_audits",
        summary_metrics: list[str] | None = None,
    ):
        super().__init__([DataKind.WEIGHTS, DataKind.WEIGHT_DIFF, DataKind.FL_MODEL])
        if type(start_round) is not int or start_round < 0 or type(frequency) is not int or frequency < 1:
            raise ValueError("start_round must be nonnegative and frequency positive")
        if Path(report_dir).is_absolute() or ".." in Path(report_dir).parts:
            raise ValueError("report_dir must be relative to the job workspace")
        self.auditor_id = auditor_id
        self.snapshot_id = snapshot_id
        self.start_round = start_round
        self.frequency = frequency
        self.report_dir = report_dir
        self.summary_metrics = summary_metrics or []
        self._reporter = MemorizationAuditReporter(report_dir, summary_metrics)

    def process(self, shareable, fl_ctx):
        try:
            # DXOFilter bypasses failed tasks, empty data and excluded kinds. Each
            # still ends this task's audit observation and must have a status.
            report = {"status": "unsupported", "round": None, "metrics": {}}
            if shareable.get_return_code() != ReturnCode.OK:
                report.update(status="skipped", reason="task_result_not_ok")
            else:
                try:
                    dxo = from_shareable(shareable)
                except ValueError:
                    report["reason"] = "non_model_result"
                else:
                    params = dxo.data.get("params") if dxo.data_kind == DataKind.FL_MODEL else dxo.data
                    if dxo.data_kind == DataKind.COLLECTION:
                        report["reason"] = "model_collection"
                    elif dxo.data_kind not in self.data_kinds:
                        report["reason"] = "unsupported_data_kind"
                    elif params is None or (isinstance(params, dict) and not params):
                        report["reason"] = "missing_parameters"
                    else:
                        return super().process(shareable, fl_ctx)
            self._report(report, fl_ctx)
            return shareable
        finally:
            # Conversion, scheduling and reporting may all bypass process_model.
            if self.snapshot_id:
                snapshot = fl_ctx.get_engine().get_component(self.snapshot_id)
                if isinstance(snapshot, AuditModelSnapshot):
                    snapshot.clear()

    def process_dxo(self, dxo, shareable, fl_ctx):
        try:
            return super().process_dxo(dxo, shareable, fl_ctx)
        except Exception as exc:
            self._report({"status": "failed", "reason": type(exc).__name__, "round": None, "metrics": {}}, fl_ctx)
            return None

    def process_model(self, model, fl_ctx):
        report = {"round": model.current_round, "status": "skipped", "metrics": {}}
        begin = time.monotonic()
        try:
            baseline = None
            if self.snapshot_id:
                snapshot = fl_ctx.get_engine().get_component(self.snapshot_id)
                if not isinstance(snapshot, AuditModelSnapshot):
                    raise TypeError("snapshot_id must name an AuditModelSnapshot")
                baseline = snapshot.pop(model, fl_ctx)
            if model.current_round is None:
                raise UnsupportedAudit("missing_round")
            if model.current_round < self.start_round or (model.current_round - self.start_round) % self.frequency:
                report["reason"] = "schedule"
            elif model.params is None:
                raise UnsupportedAudit("missing_parameters")
            elif self.snapshot_id and baseline is None:
                raise UnsupportedAudit("missing_matching_baseline")
            else:
                auditor = fl_ctx.get_engine().get_component(self.auditor_id)
                if not isinstance(auditor, MemorizationAuditor):
                    raise TypeError("auditor_id must name a MemorizationAuditor")
                result = auditor.audit(model, baseline, fl_ctx)
                json.dumps(result, allow_nan=False)
                report.update(result)
                report["status"] = "completed"
        except UnsupportedAudit as exc:
            report.update(status="unsupported", reason=str(exc), metrics={})
        except Exception as exc:
            report.update(status="failed", reason=type(exc).__name__, metrics={})
        report["elapsed_seconds"] = time.monotonic() - begin
        self._report(report, fl_ctx)
        return model

    def _report(self, report, fl_ctx):
        self._reporter.report(report, fl_ctx)
