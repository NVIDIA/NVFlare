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

"""Record bounded evaluation observations without retaining input or result data."""

from contextlib import contextmanager

from nvflare.fuel.sec import task_audit


class _Evaluation:
    def __init__(self, name, context):
        self.name = name
        self.context = context

    def record(self, outcome, *, stage="", reason="", status=""):
        # Context preparation already reported its own failure.
        if self.context is not None:
            task_audit.record(
                f"evaluation.{self.name}.{outcome}",
                context=self.context,
                stage=stage,
                reason=reason,
                status=status,
            )

    @contextmanager
    def stage(self, name, *, report_return=True):
        try:
            yield
        except Exception:
            self.record("stage_error", stage=name)
            raise
        else:
            if report_return:
                self.record("stage_returned", stage=name)

    def call(self, stage, function, /, *args, **kwargs):
        with self.stage(stage):
            return function(*args, **kwargs)


@contextmanager
def operation(name, fl_ctx):
    """Collect observations for one selected evaluation operation."""
    with task_audit.collect():
        audit = _Evaluation(name, task_audit.capture_operation_context(fl_ctx))
        audit.record("started")
        try:
            yield audit
        except Exception:
            audit.record("error")
            raise
