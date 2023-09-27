from typing import Tuple

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey


class CustomSecurityHandler(FLComponent):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.AUTHORIZE_COMMAND_CHECK:
            result, reason = self.authorize(fl_ctx=fl_ctx)
            if not result:
                fl_ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, False, sticky=False)
                fl_ctx.set_prop(FLContextKey.AUTHORIZATION_REASON, reason, sticky=False)

    def authorize(self, fl_ctx: FLContext) -> Tuple[bool, str]:
        command = fl_ctx.get_prop(FLContextKey.COMMAND_NAME)
        if command in ["check_resources"]:
            security_items = fl_ctx.get_prop(FLContextKey.SECURITY_ITEMS)
            job_meta = security_items.get(FLContextKey.JOB_META)
            if job_meta.get(JobMetaKey.JOB_NAME) == "FL Demo Job1":
                return False, f"Not authorized to execute: {command}"
            else:
                return True, ""
        else:
            return True, ""
