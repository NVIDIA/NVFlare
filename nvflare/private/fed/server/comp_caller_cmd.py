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

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.security.security import Action, FLAuthzContext
from nvflare.widgets.comp_caller import ComponentCaller
from nvflare.widgets.widget import WidgetID

from .cmd_utils import CommandUtil


class ComponentCallerCommandModule(CommandModule, CommandUtil):

    _CONN_KEY_CALLER = "caller"
    _CONN_KEY_TARGET = "comp_target"
    _CONN_KEY_ACTION = "action"
    _CONN_KEY_PARAMS = "params"

    def get_spec(self):
        return CommandModuleSpec(
            name="admin",
            cmd_specs=[
                CommandSpec(
                    name="call",
                    description="issue a call to components",
                    usage="call comp_target action params",
                    handler_func=self.call_component,
                    authz_func=self.authorize_call_component,
                    visible=True,
                )
            ],
        )

    def authorize_call_component(self, conn: Connection, args: [str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        caller = engine.get_widget(WidgetID.COMPONENT_CALLER)
        if not caller:
            conn.append_error("component caller not available")
            return False, None

        if not isinstance(caller, ComponentCaller):
            conn.append_error("system error: component caller not right object")
            return False, None

        conn.set_prop(self._CONN_KEY_CALLER, caller)

        run_info = engine.get_app_run_info()
        if not run_info or run_info.job_id < 0:
            conn.append_string("App is not running")
            return False, None

        # validate the command
        if len(args) < 3:
            conn.append_error("Syntax error. Usage: {} comp_target action [params...]".format(args[0]))
            return False, None

        comp_target = args[1]
        action = args[2]
        param_args = args[3:]

        # parse params
        params_dict = {}
        if param_args:
            for a in param_args:
                parts = a.split()
                if len(parts) <= 0:
                    conn.append_error("missing params")
                    return False, None

                # each param part must be: key=value
                for p in parts:
                    kvs = p.split("=")
                    if len(kvs) != 2:
                        conn.append_error("Syntax error: params must be key/value pairs separated by =")
                        return False, None
                    params_dict[kvs[0]] = kvs[1]

        conn.set_prop(self._CONN_KEY_ACTION, action)
        conn.set_prop(self._CONN_KEY_TARGET, comp_target)
        conn.set_prop(self._CONN_KEY_PARAMS, params_dict)

        return True, FLAuthzContext.new_authz_context(site_names=["server"], actions=[Action.TRAIN])

    def call_component(self, conn: Connection, args: [str]):
        # only support server side for now
        caller = conn.get_prop(self._CONN_KEY_CALLER)
        if not isinstance(caller, ComponentCaller):
            raise TypeError("caller must be ComponentCaller but got {}".format(type(caller)))
        action = conn.get_prop(self._CONN_KEY_ACTION)
        comp_target = conn.get_prop(self._CONN_KEY_TARGET)
        call_params = conn.get_prop(self._CONN_KEY_PARAMS)

        result = caller.call_components(target=comp_target, action=action, params=call_params)

        if not result:
            conn.append_string("No result: no component responded to the call")
            return

        # the result is a dict of: target => response
        table = conn.append_table(["Component", "Response"])
        for comp_name, resp in result.items():
            table.add_row([comp_name, resp])
