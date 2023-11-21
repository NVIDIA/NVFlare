# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotReadyToEndRun
from nvflare.fuel.utils.config_service import ConfigService


class TBI(FLComponent):

    """(TBI) Task Based Interaction is the base class for ServerRunner and ClientRunner that implement details of
    task based interactions.

    TBI implements common behavior of ServerRunner and ClientRunner.
    """

    def __init__(self):
        super().__init__()

    def _any_component_is_not_ready(self, fl_ctx: FLContext) -> bool:
        any_component_not_ready = fl_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN, False)

        if any_component_not_ready:
            self.log_debug(fl_ctx, "NOT_READY_TO_END_RUN property is set")
            return True

        # check any one has raised NotReadyToEndRun exception
        exceptions = fl_ctx.get_prop(FLContextKey.EXCEPTIONS)
        if isinstance(exceptions, dict):
            for handler_name, ex in exceptions.items():
                if isinstance(ex, NotReadyToEndRun):
                    self.log_debug(fl_ctx, f"component {handler_name} is not ready to end run")
                    return True
        return False

    def check_end_run_readiness(self, fl_ctx: FLContext):
        """Check with all components for their readiness to end run

        Args:
            fl_ctx: the FL context

        Returns:

        """
        default_max_wait = 5.0
        default_check_interval = 0.5

        # use ConfigService to determine max wait time for readiness check:
        #   the job config could define variable "end_run_readiness_timeout"
        #   the user could define OS env var NVFLARE_END_RUN_READINESS_TIMEOUT
        max_wait = ConfigService.get_float_var(
            name="end_run_readiness_timeout", conf=SystemConfigs.APPLICATION_CONF, default=default_max_wait
        )
        if max_wait <= 0:
            max_wait = default_max_wait

        # use ConfigService to determine interval for readiness check:
        #   the job config could define variable "end_run_readiness_check_interval"
        #   the user could define OS env var NVFLARE_END_RUN_READINESS_CHECK_INTERVAL
        check_interval = ConfigService.get_float_var(
            name="end_run_readiness_check_interval", conf=SystemConfigs.APPLICATION_CONF, default=default_check_interval
        )
        if check_interval <= 0:
            check_interval = default_check_interval

        self.log_debug(fl_ctx, f"=== end_run_readiness: {max_wait=} {check_interval=}")
        check_start_time = time.time()
        while True:
            fl_ctx.remove_prop(FLContextKey.NOT_READY_TO_END_RUN, force_removal=True)
            fl_ctx.remove_prop(FLContextKey.EXCEPTIONS, force_removal=True)

            self.log_info(fl_ctx, "Firing CHECK_END_RUN_READINESS ...")
            self.fire_event(EventType.CHECK_END_RUN_READINESS, fl_ctx)

            if self._any_component_is_not_ready(fl_ctx):
                if time.time() - check_start_time > max_wait:
                    # we have waited too long
                    self.log_warning(fl_ctx, f"quit waiting for component ready-to-end-run after {max_wait} seconds")
                    return
                else:
                    time.sleep(check_interval)
            else:
                # all components are ready to end
                return
