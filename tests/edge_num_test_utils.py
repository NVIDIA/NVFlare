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

"""In-process numeric edge workflow helpers; only logging is mocked."""

from unittest.mock import Mock

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.edge.aggregators.num_dxo_factory import NumDXOAggrFactory
from nvflare.edge.assessors.async_num import AsyncNumAssessor
from nvflare.edge.mud import Device, ModelUpdate, StateUpdateReport
from nvflare.edge.updaters.emd import EdgeModelUpdater


def quiet(component):
    for name in ("log_debug", "log_info", "log_warning", "log_error"):
        setattr(component, name, Mock())
    return component


def numeric_update(version, devices, value, count=1, timestamp=1.0, kind="number"):
    return ModelUpdate(
        version,
        DXO(kind, {"value": value, "count": count}).to_shareable(),
        {device: timestamp for device in devices},
    )


def report(*updates):
    return StateUpdateReport(1, 1, {mu.model_version: mu for mu in updates}, {}).to_shareable()


def make_assessor(ctx, threshold=2, history=3, device_reuse=False):
    assessor = quiet(AsyncNumAssessor(threshold, 10, history, 2, device_reuse=device_reuse))
    assessor.start_task(ctx)
    devices = {d: Device(d, "site", 1.0) for d in ("A", "B")}
    assessor.process_child_update(StateUpdateReport(0, 0, None, devices).to_shareable(), ctx)
    return assessor


def make_workflow(history=None):
    ctx = FLContext()
    assessor = make_assessor(ctx, device_reuse=True)
    updater = quiet(EdgeModelUpdater(NumDXOAggrFactory(), history))
    updater.start_task(assessor.start_task(ctx), ctx)
    return assessor, updater, ctx


def forward(updater, assessor, ctx):
    update = updater.prepare_update_for_parent(ctx)
    accepted, reply = assessor.process_child_update(update, ctx)
    updater.process_parent_update_reply(reply, ctx)
    return accepted, StateUpdateReport.from_shareable(update)
