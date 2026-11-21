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
from types import SimpleNamespace

import numpy as np
import pytest

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.fuel.sec.audit import Auditor, AuditService


@pytest.mark.parametrize(
    "case, read_outcome, value_returned",
    [
        ("absolute", "stage_returned", True),
        ("relative", "stage_returned", True),
        ("diagnostic_error_after_successful_read", "stage_returned", True),
        ("diagnostic_error_after_failed_read", "stage_error", False),
        ("missing", "stage_error", False),
        ("corrupt", "stage_error", False),
        ("unknown", "", False),
        ("preparation_error", "", False),
    ],
)
def test_numpy_locator_records_actual_read_and_preserves_result(
    tmp_path, monkeypatch, case, read_outcome, value_returned
):
    audit_path = tmp_path / "audit.jsonl"
    writer = Auditor(str(audit_path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "private-checkpoint.npy"
    model_name = "private-model-name"
    weights = np.array([73, 81], dtype=np.int64)

    if value_returned:
        np.save(model_path, weights, allow_pickle=False)
    elif case == "corrupt":
        model_path.write_bytes(b"private-invalid-file-contents")

    def get_run_dir(job_id):
        assert job_id == "job-1"
        if case == "preparation_error":
            raise RuntimeError("private-preparation-error")
        return str(tmp_path)

    workspace = SimpleNamespace(get_run_dir=get_run_dir)
    engine = SimpleNamespace(get_workspace=lambda: workspace, fire_event=lambda event_type, fl_ctx: None)
    ctx = FLContextManager(engine=engine, identity_name="server", job_id="job-1").new_context()
    ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "evaluation scope", private=True, sticky=False)
    filename = model_path.name if case in ("relative", "preparation_error") else str(model_path)
    locator = NPModelLocator(model_name={model_name: filename})

    diagnostic_error = RuntimeError("private-diagnostic-error")

    def fail_logging(*args, **kwargs):
        raise diagnostic_error

    if case == "diagnostic_error_after_successful_read":
        monkeypatch.setattr(locator, "log_info", fail_logging)

    try:
        if case == "diagnostic_error_after_failed_read":
            monkeypatch.setattr(locator, "log_error", fail_logging)
            monkeypatch.setattr(locator, "log_exception", fail_logging)
            with pytest.raises(RuntimeError) as caught:
                locator.locate_model(model_name, ctx)
            assert caught.value is diagnostic_error
        else:
            result = locator.locate_model("private-unmapped-name" if case == "unknown" else model_name, ctx)
            if value_returned:
                assert result.data_kind == DataKind.WEIGHTS
                np.testing.assert_array_equal(result.data[NPConstants.NUMPY_KEY], weights)
                assert result.meta == {}
            else:
                assert result is None
    finally:
        writer.close()

    raw_audit = audit_path.read_text()
    records = [json.loads(line) for line in raw_audit.splitlines()]
    expected = [("model.retrieve.started", "")]
    if read_outcome:
        expected.append((f"model.retrieve.{read_outcome}", "stage=file_read"))
    if case == "diagnostic_error_after_failed_read":
        expected.append(("model.retrieve.error", ""))
    else:
        status = "value_returned" if value_returned else "no_value"
        expected.append(("model.retrieve.returned", f"status={status}"))
    assert [(record["action"], record.get("message", "")) for record in records] == expected
    for record in records:
        assert set(record) <= {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
        assert all(isinstance(value, str) for value in record.values())
        assert record["user"] == "unknown"
        assert record["job"] == "job-1"
        assert record["scope"] == "evaluation scope"
    for private in (
        model_name,
        model_path.name,
        str(tmp_path),
        "private-unmapped-name",
        "private-invalid-file-contents",
        "private-diagnostic-error",
        "private-preparation-error",
    ):
        assert private not in raw_audit
