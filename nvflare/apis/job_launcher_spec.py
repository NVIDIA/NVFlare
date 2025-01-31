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
from abc import abstractmethod

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.common.exit_codes import ProcessExitCode


class JobProcessArgs:

    EXE_MODULE = "exe_module"
    WORKSPACE = "workspace"
    STARTUP_DIR = "startup_dir"
    APP_ROOT = "app_root"
    AUTH_TOKEN = "auth_token"
    TOKEN_SIGNATURE = "auth_signature"
    SSID = "ssid"
    JOB_ID = "job_id"
    CLIENT_NAME = "client_name"
    ROOT_URL = "root_url"
    PARENT_URL = "parent_url"
    PARENT_CONN_SEC = "parent_conn_sec"
    SERVICE_HOST = "service_host"
    SERVICE_PORT = "service_port"
    HA_MODE = "ha_mode"
    TARGET = "target"
    SCHEME = "scheme"
    STARTUP_CONFIG_FILE = "startup_config_file"
    RESTORE_SNAPSHOT = "restore_snapshot"
    OPTIONS = "options"


class JobReturnCode(ProcessExitCode):
    SUCCESS = 0
    EXECUTION_ERROR = 1
    ABORTED = 9
    UNKNOWN = 127


def add_launcher(launcher, fl_ctx: FLContext):
    job_launcher: list = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER, [])
    job_launcher.append(launcher)
    fl_ctx.set_prop(FLContextKey.JOB_LAUNCHER, job_launcher, private=True, sticky=False)


class JobHandleSpec:
    @abstractmethod
    def terminate(self):
        """To terminate the job run.

        Returns: the job run return code.

        """
        raise NotImplementedError()

    @abstractmethod
    def poll(self):
        """To get the return code of the job run.

        Returns: return_code

        """
        raise NotImplementedError()

    @abstractmethod
    def wait(self):
        """To wait until the job run complete.

        Returns: returns until the job run complete.

        """
        raise NotImplementedError()


class JobLauncherSpec(FLComponent):
    @abstractmethod
    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        """To launch a job run.

        Args:
            job_meta: job metadata
            fl_ctx: FLContext

        Returns: boolean to indicates the job launch success or fail.

        """
        raise NotImplementedError()
