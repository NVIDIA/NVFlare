from .fl_context import FLContext
from .workspace import Workspace


class AppDeployerSpec(object):
    def deploy(
        self, workspace: Workspace, job_id: str, job_meta: dict, app_name: str, app_data: bytes, fl_ctx: FLContext
    ) -> str:
        pass
