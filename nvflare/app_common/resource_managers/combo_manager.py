# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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


import uuid
from typing import Tuple

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec


class ComboManager(ResourceManagerSpec, FLComponent):
    def __init__(self, manager_ids: [str]):
        FLComponent.__init__(self)
        self.manager_ids = manager_ids
        self.managers = []
        self.combo_token = None
        self.tokens = []

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            engine = fl_ctx.get_engine()
            for mgr_id in self.manager_ids:
                mgr = engine.get_component(mgr_id)
                if not mgr:
                    self.system_panic(reason=f"cannot find component '{mgr_id}'", fl_ctx=fl_ctx)
                    return
                if not isinstance(mgr, ResourceManagerSpec):
                    self.system_panic(
                        reason=f"component '{mgr_id}' must be ResourceManagerSpec but got {type(mgr)}", fl_ctx=fl_ctx
                    )
                    return
                self.managers.append(mgr)

    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> Tuple[bool, str]:
        self.combo_token = None
        self.tokens = []
        denied = False
        for m in self.managers:
            ok, t = m.check_resources(resource_requirement, fl_ctx)
            if ok:
                if t:
                    self.tokens.append((m, t))
            else:
                denied = True
                break

        if self.tokens:
            self.combo_token = str(uuid.uuid4())
        else:
            self.combo_token = ""

        if denied:
            if self.tokens:
                self.cancel_resources(resource_requirement, self.combo_token, fl_ctx)
            return False, ""
        else:
            return True, self.combo_token

    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        if token != self.combo_token:
            return

        self.combo_token = None
        if self.tokens:
            for mgr, t in self.tokens:
                assert isinstance(mgr, ResourceManagerSpec)
                mgr.cancel_resources(resource_requirement, t, fl_ctx)
            self.tokens = None

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        resources = {}
        if token == self.combo_token and self.tokens:
            for mgr, t in self.tokens:
                assert isinstance(mgr, ResourceManagerSpec)
                r = mgr.allocate_resources(resource_requirement, t, fl_ctx)
                resources.update(r)
        return resources

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        if token == self.combo_token and self.tokens:
            for mgr, t in self.tokens:
                assert isinstance(mgr, ResourceManagerSpec)
                mgr.free_resources(resources, t, fl_ctx)

    def report_resources(self, fl_ctx):
        report = {}
        if self.tokens:
            for mgr, t in self.tokens:
                assert isinstance(mgr, ResourceManagerSpec)
                r = mgr.report_resources(fl_ctx)
                report.update(r)
        return report
