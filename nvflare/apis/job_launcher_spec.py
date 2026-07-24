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
import os
from abc import ABC, abstractmethod

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
    TARGET = "target"
    SCHEME = "scheme"
    STARTUP_CONFIG_FILE = "startup_config_file"
    RESTORE_SNAPSHOT = "restore_snapshot"
    OPTIONS = "options"


class JobProcessEnv:
    """Env var names for job-process bootstrap credentials; the CJ/SJ arg parsers consume
    and remove them (a CLI-supplied value wins)."""

    AUTH_TOKEN = "NVFLARE_JOB_AUTH_TOKEN"
    TOKEN_SIGNATURE = "NVFLARE_JOB_TOKEN_SIGNATURE"
    SSID = "NVFLARE_JOB_SSID"


# The complete bootstrap-credential set; every scrub of job-process credentials
# must consume this so a future credential cannot be missed by one of them.
CREDENTIAL_ENV_NAMES = (JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID)


def pop_credential_env() -> dict:
    """Remove every JobProcessEnv credential from the environment and return it.

    Called at job-process arg parsing; removing all of them — parsed or not — keeps
    job-spawned children from inheriting credentials. Empty values count as absent so
    a blank env var fails parsing like a missing one.
    """
    return {name: os.environ.pop(name, None) or None for name in CREDENTIAL_ENV_NAMES}


class JobReturnCode(ProcessExitCode):
    SUCCESS = 0
    EXECUTION_ERROR = 1
    ABORTED = 9
    UNKNOWN = 127


def add_launcher(launcher, fl_ctx: FLContext):
    job_launcher: list = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER, [])
    job_launcher.append(launcher)
    fl_ctx.set_prop(FLContextKey.JOB_LAUNCHER, job_launcher, private=True, sticky=False)


class JobHandleSpec(ABC):
    @abstractmethod
    def terminate(self):
        """To terminate the job run.

        Returns: None

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


class JobLauncherSpec(FLComponent, ABC):
    @abstractmethod
    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        """To launch a job run.

        Args:
            job_meta: job metadata
            fl_ctx: FLContext

        Returns: a JobHandle instance.

        """
        raise NotImplementedError()
