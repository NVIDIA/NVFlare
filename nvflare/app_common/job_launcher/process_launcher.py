# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from abc import abstractmethod
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import get_job_meta_study
from nvflare.apis.job_launcher_spec import (
    JobHandleSpec,
    JobLauncherSpec,
    JobReturnCode,
    add_launcher,
    publish_job_handle,
)
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.study_runtime import (
    load_study_runtime_file,
    resolve_study_runtime,
    study_runtime_file_path,
)
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path, get_credential_env
from nvflare.utils.process_utils import (
    ProcessAdapter,
    ProcessSpawnError,
    linux_process_group_has_live_members,
    spawn_process,
)

JOB_RETURN_CODE_MAPPING = {0: JobReturnCode.SUCCESS, 1: JobReturnCode.EXECUTION_ERROR, 9: JobReturnCode.ABORTED}


class ProcessHandle(JobHandleSpec):
    def __init__(
        self,
        process: Optional[subprocess.Popen] = None,
        pid: Optional[int] = None,
        process_adapter: Optional[ProcessAdapter] = None,
        owns_process_group: bool = False,
        app_dir: Optional[str] = None,
    ):
        super().__init__()
        self.owns_process_group = owns_process_group
        self.app_dir = app_dir
        self._group_exit_confirmed = False

        if process_adapter:
            self.adapter = process_adapter
        elif process or pid is not None:
            self.adapter = ProcessAdapter(process=process, pid=pid)
        else:
            raise ValueError("ProcessHandle requires a process object, a pid, or a ProcessAdapter.")

    def terminate(self):
        if self._group_exit_confirmed:
            return
        if not self.owns_process_group:
            self.adapter.terminate()
            return
        try:
            # The leader may already be reaped while its owned group still exists.
            os.killpg(self.adapter.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except (OSError, AttributeError) as e:
            logging.getLogger(__name__).warning("Could not terminate process group %s: %s", self.adapter.pid, e)

    def poll(self):
        code = self.adapter.poll()
        if code is None:
            return JobReturnCode.UNKNOWN
        return JOB_RETURN_CODE_MAPPING.get(code, JobReturnCode.EXECUTION_ERROR)

    def wait(self):
        self.adapter.wait()
        if self.owns_process_group:
            # A successful leader can also leave writers behind. Each grace
            # period bounds a cancellation attempt, never ownership retention.
            while not self._wait_for_group_exit():
                self.terminate()

    def _wait_for_group_exit(self):
        deadline = time.monotonic() + 10.0
        while not self._group_exited():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))
        return True

    def can_restart(self):
        if not self._group_exited():
            return False
        if self.app_dir is None:
            return True
        try:
            # Written before Popen and cleared only after detached-trainer exit.
            # Its presence makes CJ group exit insufficient evidence for recovery.
            return not any(
                name.startswith("client_api_bootstrap_") and name.endswith(".json") for name in os.listdir(self.app_dir)
            )
        except OSError:
            return False

    def _group_exited(self):
        if self._group_exit_confirmed:
            return True
        # The launcher starts a new session. Other supplied processes need not
        # own their group.
        if not self.owns_process_group or self.adapter.poll() is None:
            return False
        try:
            os.killpg(self.adapter.pid, 0)
        except ProcessLookupError:
            self._group_exit_confirmed = True
            return True
        except (OSError, AttributeError):
            return False
        if sys.platform == "linux":
            # Zombies retain the PGID but cannot execute or write. Inspect
            # threads too: a zombie main thread can have live worker threads.
            # Require two complete observations to cover a concurrent fork.
            if (
                linux_process_group_has_live_members(self.adapter.pid) is False
                and linux_process_group_has_live_members(self.adapter.pid) is False
            ):
                self._group_exit_confirmed = True
                return True
        return False


class ProcessJobLauncher(JobLauncherSpec):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:

        new_env = os.environ.copy()
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        runtime_file = study_runtime_file_path(workspace_obj.get_root_dir())
        if os.path.exists(runtime_file):
            runtime_map = load_study_runtime_file(runtime_file, launcher_mode="process", logger=self.logger)
            study = get_job_meta_study(job_meta)
            runtime = resolve_study_runtime(runtime_map, study, runtime_file, logger=self.logger)
            for ref in runtime.secret_env:
                value = new_env.get(ref.source)
                if value is None:
                    raise ValueError(
                        f"secret_env '{ref.name}' for study '{study}' requires env var "
                        f"'{ref.source}' in the launcher environment, but it is not set"
                    )
                runtime.env[ref.name] = value
            new_env.update(runtime.env)

        job_id = job_meta.get(JobConstants.JOB_ID)
        app_custom_folder = workspace_obj.get_app_custom_dir(job_id)
        if app_custom_folder != "":
            add_custom_dir_to_path(app_custom_folder, new_env)

        new_env.update(get_credential_env(fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS) or {}))

        command = self.get_command(job_meta, fl_ctx)
        argv = shlex.split(command, True)

        # Use the spawn_process utility which handles the choice between posix_spawn and subprocess
        app_dir = workspace_obj.get_app_dir(job_id)
        try:
            process_adapter = spawn_process(argv, new_env)
        except ProcessSpawnError as error:
            handle = ProcessHandle(process_adapter=error.adapter, owns_process_group=True, app_dir=app_dir)
            publish_job_handle(handle, fl_ctx)
            raise
        handle = ProcessHandle(process_adapter=process_adapter, owns_process_group=True, app_dir=app_dir)
        publish_job_handle(handle, fl_ctx)
        return handle

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_JOB_LAUNCH:
            add_launcher(self, fl_ctx)

    @abstractmethod
    def get_command(self, job_meta, fl_ctx) -> str:
        """To generate the command to launcher the job in sub-process

        Args:
            fl_ctx: FLContext
            job_meta: job meta data

        Returns:
            launch command

        """
        pass
