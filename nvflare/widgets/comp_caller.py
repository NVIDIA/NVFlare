# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext

from .widget import Widget


class CallInfo(object):
    def __init__(self, target: str, action: str, params: dict):
        self.target = target  # target component(s) that the call is applied to
        self.action = action  # action of the call
        self.params = params  # params of the call
        self.results = {}  # results of components that tried to apply the params

    def report_result(self, target: str, result: str = "OK"):
        self.results[target] = result


class ComponentCaller(Widget):

    EVENT_TYPE_CALL_COMPONENT = "comp_caller.call"
    CTX_KEY_CALL_INFO = "comp_caller.call_info"

    def __init__(self):
        Widget.__init__(self)
        self.engine = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
        elif event_type == EventType.END_RUN:
            self.engine = None

    def call_components(self, target: str, action: str, params: dict):
        """
        Make a call to component(s).

        Args:
            target: the target spec of the component(s) to be called.
            action: action of the call
            params: parameters for the call

        Returns: None or dict of result: comp name => result string

        NOTE: each component that wants to participate the call mechanism must:

        - Listen to the event EVENT_TYPE_CALL_COMPONENT

        - In the event handler, decide whether the call is applicable to it by comparing itself to
        the 'target'. The target could be a specific component ID, or a type of components

        - decide further whether the call is applicable to it by looking at the 'action'.
        Conceptually, the action is like a function to be called on the component.
        If the component doesn't support the action, simply ignore the call.

        - if the call is applicable, always report the execution status to the call.

        """

        # NOTE: it's important to assign self.engine to a new var!
        # This is because another thread may fire the END_RUN event, which will cause
        # self.engine to be set to None, just after checking it being None and before using it!
        engine = self.engine
        if not engine:
            return None

        # NOTE: we need a new context here to make sure all sticky props are copied!
        with engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            info = CallInfo(target=target, action=action, params=params)
            fl_ctx.set_prop(key=self.CTX_KEY_CALL_INFO, value=info, sticky=False, private=True)

            engine.fire_event(event_type=self.EVENT_TYPE_CALL_COMPONENT, fl_ctx=fl_ctx)

            return info.results
