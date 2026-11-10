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
from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_common.abstract.memorization_auditor import MemorizationAuditor, UnsupportedAudit
from nvflare.app_common.filters.model_filter import ModelFilter
from nvflare.app_common.widgets.memorization_audit_reporter import MemorizationAuditReporter
from nvflare.fuel.utils.experimental import experimental

# Private/non-sticky props still travel to MultiProcessExecutor workers. Keep the
# snapshot outside props: context relays and the FOBS ContextDecomposer omit it.
# Both filters must run on the original task context in the same process.
_BASELINE_ATTR = "_nvflare_memorization_audit_baseline"


def _validate_schedule(start_round, frequency):
    if type(start_round) is not int or start_round < 0 or type(frequency) is not int or frequency < 1:
        raise ValueError("start_round must be nonnegative and frequency positive")


def _audit_due(round_number, start_round, frequency):
    return round_number >= start_round and (round_number - start_round) % frequency == 0


class AuditSnapshotFilter(ModelFilter):
    """Copy the incoming full model into an attribute on the original task context.

    Place after input transformations. The paired output filter consumes this
    process-local baseline from the same FLContext object. It is not a context
    property: worker relays, context clones and the FOBS context decomposer omit it.
    Neither filter retains it between tasks.
    Match start_round/frequency to the output filter to avoid copies on skipped rounds.
    """

    @experimental("AuditSnapshotFilter API may change")
    def __init__(self, start_round: int = 0, frequency: int = 1):
        super().__init__([DataKind.WEIGHTS, DataKind.FL_MODEL])
        _validate_schedule(start_round, frequency)
        self.start_round = start_round
        self.frequency = frequency

    def process_model(self, model, fl_ctx):
        if model.params_type != ParamsType.FULL or model.current_round is None:
            raise UnsupportedAudit("Baseline requires full parameters and a round identity")
        if _audit_due(model.current_round, self.start_round, self.frequency):
            setattr(fl_ctx, _BASELINE_ATTR, deepcopy(model))
        return None

    def process(self, shareable, fl_ctx):
        fl_ctx.__dict__.pop(_BASELINE_ATTR, None)
        try:
            if from_shareable(shareable).data_kind == DataKind.COLLECTION:
                raise UnsupportedAudit("Audit snapshot requires a single model")
            return super().process(shareable, fl_ctx)
        except Exception as exc:
            fl_ctx.__dict__.pop(_BASELINE_ATTR, None)
            self.log_warning(fl_ctx, f"Audit baseline capture failed: {type(exc).__name__}")
            return shareable


class MemorizationAuditFilter(ModelFilter):
    """Experimental, report-only outgoing model audit. Never suppresses an update.

    Pass an auditor instance and install AuditSnapshotFilter on task input. The
    baseline lives in an attribute on the original task's FLContext, outside its
    serializable properties. Both filters must use that same object in the same
    process. Job export rebuilds the auditor as a nested object; it does not
    receive component events. Reports are local
    JSON under the job workspace. Only explicitly selected scalar metrics
    are emitted to analytics; detailed audit data belongs in the auditor's local
    storage. Runtime failures log a failed report and leave training unaffected.
    """

    @experimental("MemorizationAuditFilter API may change")
    def __init__(
        self,
        auditor: MemorizationAuditor,
        start_round: int = 0,
        frequency: int = 1,
        report_dir: str = "memorization_audits",
        summary_metrics: list[str] | None = None,
    ):
        super().__init__([DataKind.WEIGHTS, DataKind.WEIGHT_DIFF, DataKind.FL_MODEL])
        _validate_schedule(start_round, frequency)
        if Path(report_dir).is_absolute() or ".." in Path(report_dir).parts:
            raise ValueError("report_dir must be relative to the job workspace")
        if not isinstance(auditor, MemorizationAuditor):
            raise TypeError("auditor must be a MemorizationAuditor")
        self.auditor = auditor
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
                except Exception as exc:
                    report.update(status="failed", reason=type(exc).__name__)
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
            fl_ctx.__dict__.pop(_BASELINE_ATTR, None)

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
            baseline = fl_ctx.__dict__.pop(_BASELINE_ATTR, None)
            if model.current_round is None:
                raise UnsupportedAudit("missing_round")
            if not _audit_due(model.current_round, self.start_round, self.frequency):
                report["reason"] = "schedule"
            elif model.params is None:
                raise UnsupportedAudit("missing_parameters")
            elif baseline is None:
                raise UnsupportedAudit("missing_baseline")
            elif baseline.current_round != model.current_round:
                raise UnsupportedAudit("baseline_round_mismatch")
            else:
                result = self.auditor.audit(model, baseline, fl_ctx)
                json.dumps(result, allow_nan=False)
                report.update(result)
                report["status"] = "completed"
        except UnsupportedAudit as exc:
            report.update(status="unsupported", reason=str(exc), metrics={})
        except Exception as exc:
            report.update(status="failed", reason=type(exc).__name__, metrics={})
        report["elapsed_seconds"] = time.monotonic() - begin
        self._report(report, fl_ctx)
        return None

    def _report(self, report, fl_ctx):
        self._reporter.report(report, fl_ctx)
