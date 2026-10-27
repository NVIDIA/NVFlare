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
import time
from abc import abstractmethod
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import get_job_meta_study
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobReturnCode, add_launcher
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.study_runtime import (
    load_study_runtime_file,
    resolve_study_runtime,
    study_runtime_file_path,
)
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path, get_credential_env
from nvflare.utils.process_utils import ProcessAdapter, spawn_process

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

        if process_adapter:
            self.adapter = process_adapter
        elif process or pid is not None:
            self.adapter = ProcessAdapter(process=process, pid=pid)
        else:
            raise ValueError("ProcessHandle requires a process object, a pid, or a ProcessAdapter.")

    def terminate(self):
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
        if self.owns_process_group and self.adapter.poll() not in (None, 0):
            # Allow children to exit, then bound cleanup of any remaining members.
            if not self._wait_for_group_exit():
                self.terminate()
                self._wait_for_group_exit()

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
        # Disposable task workers have separate process groups. CJ group exit
        # cannot prove those scopes have settled. Their existing runtime root is
        # created before any worker launch and retained with the attempt evidence.
        # CP must not replace this CJ without an owner that can settle those groups.
        try:
            os.lstat(os.path.join(os.path.dirname(self.app_dir), ".nvflare", "task-execution"))
        except FileNotFoundError:
            pass
        except OSError:
            return False
        else:
            return False
        try:
            # Written before Popen and cleared only after detached-trainer exit.
            # Its presence makes CJ group exit insufficient evidence for recovery.
            return not any(
                name.startswith("client_api_bootstrap_") and name.endswith(".json") for name in os.listdir(self.app_dir)
            )
        except OSError:
            return False

    def _group_exited(self):
        # The launcher starts a new session. Other supplied processes need not
        # own their group.
        if not self.owns_process_group or self.adapter.poll() is None:
            return False
        try:
            os.killpg(self.adapter.pid, 0)
        except ProcessLookupError:
            return True
        except (OSError, AttributeError):
            return False
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

        self.prepare_environment(job_meta, fl_ctx, new_env)

        # Use the spawn_process utility which handles the choice between posix_spawn and subprocess
        process_adapter = spawn_process(argv, new_env)

        return ProcessHandle(
            process_adapter=process_adapter, owns_process_group=True, app_dir=workspace_obj.get_app_dir(job_id)
        )

    def prepare_environment(self, job_meta, fl_ctx, environment):
        """Apply trusted launch-specific policy after study environment overrides."""
        pass

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
