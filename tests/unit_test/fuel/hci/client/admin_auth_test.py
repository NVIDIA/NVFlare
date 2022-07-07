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
import datetime
import glob
import json
import os
import time
import uuid
from typing import List, Optional
from unittest.mock import MagicMock, Mock

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.apis.job_def import Job, JobMetaKey, RunStatus
from nvflare.fuel.hci.base64_utils import bytes_to_b64str
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.cmd_arg_utils import join_args
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule
from nvflare.fuel.hci.server.audit import CommandAudit
from nvflare.fuel.hci.server.authz import AuthorizationService, AuthzCommandModule, AuthzFilter
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.file_transfer import FileTransferModule
from nvflare.fuel.hci.server.login import LoginModule, SimpleAuthenticator
from nvflare.fuel.hci.server.sess import Session, SessionManager
from nvflare.fuel.hci.zip_utils import split_path, zip_directory_to_bytes
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.private.fed.app.default_app_validator import DefaultAppValidator
from nvflare.private.fed.server.app_authz import AppAuthzService
from nvflare.private.fed.server.server_cmd_modules import ServerCommandModules
from nvflare.security.security import FLAuthorizer

USER_A = "user-a@a.org"
USER_B = "user-b@b.org"
SUPER = "admin@nvflare.com"
AUTH_ERROR = "Authorization Error:"

auth_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../data/authorization"))


class MockConnection(Connection):
    def __init__(self, request):
        super().__init__(None, None)
        self.request = request
        self.result = ""

    def _send_line(self, line: str, all_end=False):
        self.result += line

    def get_result(self):
        self.flush()
        return self.result


class MockSessionManager(SessionManager):
    def __init__(self, user_name: str):
        super().__init__()
        self.user_name = user_name

    def get_session(self, token: str):
        sess = Session()
        sess.user_name = self.user_name
        sess.start_time = time.time()
        sess.last_active_time = sess.start_time
        sess.token = token
        return sess


class MockJobDefManager(SimpleJobDefManager):
    def __init__(self):
        super().__init__()
        self.jobs = self._create_jobs()

    def _get_job_store(self, fl_ctx):
        return Mock(name="job_store")

    def get_job(self, jid: str, fl_ctx: FLContext) -> Job:
        return next(job for job in self.jobs if job.job_id == jid)

    def get_all_jobs(self, fl_ctx: FLContext) -> List[Job]:
        return self.jobs

    def find_job_by_name(self, name: str) -> Optional[Job]:
        return next(job for job in self.jobs if job.meta.get(JobMetaKey.JOB_FOLDER_NAME) == name)

    @staticmethod
    def _create_jobs():
        meta_files = glob.glob(auth_root + "/jobs/*/meta.json")
        jobs = []
        for meta_file in meta_files:
            with open(meta_file) as f:
                meta = json.load(f)
                jid = str(uuid.uuid4())
                folder = os.path.dirname(meta_file)
                job_name = os.path.basename(folder)
                meta[JobMetaKey.JOB_FOLDER_NAME] = job_name
                meta[JobMetaKey.JOB_ID.value] = jid
                meta[JobMetaKey.SUBMIT_TIME.value] = time.time()
                meta[JobMetaKey.SUBMIT_TIME_ISO.value] = (
                    datetime.datetime.fromtimestamp(meta[JobMetaKey.SUBMIT_TIME]).astimezone().isoformat()
                )
                meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value
                job = Job(
                    job_id=jid,
                    resource_spec=meta.get(JobMetaKey.RESOURCE_SPEC),
                    deploy_map=meta.get(JobMetaKey.DEPLOY_MAP),
                    meta=meta,
                )

                jobs.append(job)

        return jobs


class MockAdminServer:

    job_def_manager = MockJobDefManager()

    def __init__(self, user_name: str):
        server_engine = MagicMock(name="server_engine")
        server_engine.job_def_manager = MockAdminServer.job_def_manager
        cmd_reg = new_command_register_with_builtin_module(app_ctx=server_engine)

        authenticator = SimpleAuthenticator({})
        self.sess_mgr = MockSessionManager(user_name)
        login_module = LoginModule(authenticator, self.sess_mgr)
        cmd_reg.register_module(login_module)
        cmd_reg.add_filter(login_module)

        SecurityContentService.initialize(auth_root)
        AuthorizationService.initialize(FLAuthorizer())
        authorizer = AuthorizationService.get_authorizer()
        authz_filter = AuthzFilter(authorizer=authorizer)
        cmd_reg.add_filter(authz_filter)
        authz_cmd_module = AuthzCommandModule(authorizer=authorizer)
        cmd_reg.register_module(authz_cmd_module)

        # audit filter records commands to audit trail
        AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)
        auditor = AuditService.get_auditor()
        audit_filter = CommandAudit(auditor)
        cmd_reg.add_filter(audit_filter)

        self.file_upload_dir = os.path.join(auth_root, "jobs")
        self.upload_dir = os.path.join(auth_root, "jobs")
        self.file_download_dir = auth_root

        AppAuthzService.initialize(DefaultAppValidator())
        cmd_reg.register_module(
            FileTransferModule(
                upload_dir=self.file_upload_dir,
                download_dir=self.file_download_dir,
                upload_folder_authz_func=AppAuthzService.authorize_upload,
            )
        )

        cmd_reg.register_module(self.sess_mgr)
        cmd_modules = ServerCommandModules.cmd_modules
        if cmd_modules:
            if not isinstance(cmd_modules, list):
                raise TypeError("cmd_modules must be list but got {}".format(type(cmd_modules)))

            for m in cmd_modules:
                if not isinstance(m, CommandModule):
                    raise TypeError("cmd_modules must contain CommandModule but got element of type {}".format(type(m)))
                cmd_reg.register_module(m)

        self.cmd_reg = cmd_reg
        cmd_reg.finalize()

    def shutdown(self):
        self.sess_mgr.shutdown()

    def submit_job(self, job_folder: str) -> dict:

        if job_folder.endswith("/"):
            job_folder = job_folder.rstrip("/")

        full_path = os.path.join(self.upload_dir, job_folder)
        if not os.path.isdir(full_path):
            return {"status": APIStatus.ERROR_RUNTIME, "details": f"'{full_path}' is not a valid folder."}

        # zip the data
        data = zip_directory_to_bytes(self.upload_dir, job_folder)

        job_folder = split_path(full_path)[1]
        b64str = bytes_to_b64str(data)
        parts = ["_submit_job", job_folder, b64str]
        command = join_args(parts)
        return self._do_command(command)

    def list_jobs(self, args: str = None) -> dict:
        command = "list_jobs"
        if args:
            command += " " + args
        return self._do_command(command)

    def abort_job(self, job_id: str) -> dict:
        command = "abort_job " + job_id
        return self._do_command(command)

    def _do_command(self, command: str):
        request = {"data": [{"type": "token", "data": "session-token"}, {"type": "command", "data": command}]}
        conn = MockConnection(request)
        self.cmd_reg.process_command(conn, command)
        return json.loads(conn.get_result())


class TestAdminAuth:
    @classmethod
    def setup_class(cls):
        cls.admin_server_super = MockAdminServer(SUPER)
        cls.admin_server_a = MockAdminServer(USER_A)
        cls.admin_server_b = MockAdminServer(USER_B)

    @classmethod
    def teardown_class(cls):
        cls.admin_server_b.shutdown()
        cls.admin_server_a.shutdown()
        cls.admin_server_super.shutdown()

    def test_submit_job_self(self):
        """Test TRAIN_SELF, users can only run jobs on their own sites"""

        response = TestAdminAuth.admin_server_a.submit_job("job_a")
        assert self._find_type(response, "success")

        response = TestAdminAuth.admin_server_b.submit_job("job_b")
        assert self._find_type(response, "success")

    def test_submit_job_other(self):
        """Test negative TRAIN_SELF, users can not run jobs on other's sites"""

        response = TestAdminAuth.admin_server_a.submit_job("job_b")
        assert AUTH_ERROR in self._get_type(response, "error")

        response = TestAdminAuth.admin_server_b.submit_job("job_a")
        assert AUTH_ERROR in self._get_type(response, "error")

    def test_submit_job_super(self):
        """Test TRAIN_ALL, superusers can run jobs on all sites"""

        response = TestAdminAuth.admin_server_super.submit_job("job_a")
        assert self._find_type(response, "success")

        response = TestAdminAuth.admin_server_super.submit_job("job_b")
        assert self._find_type(response, "success")

    def test_list_jobs_regular(self):
        """Test VIEW_SELF, users can only see jobs on their own sites"""

        response = TestAdminAuth.admin_server_a.list_jobs()
        data = self._get_type(response, "string")
        count = data.count("SUBMITTED")
        assert count == 1, f"More jobs listed than expected: {count}"

        response = TestAdminAuth.admin_server_b.list_jobs()
        data = self._get_type(response, "string")
        count = data.count("SUBMITTED")
        assert count == 1, f"More jobs listed than expected: {count}"

    def test_list_jobs_super(self):
        """Test VIEW_ALL, superusers can see all jobs"""

        response = TestAdminAuth.admin_server_super.list_jobs()
        data = self._get_type(response, "string")
        count = data.count("SUBMITTED")
        assert count == 2, f"Not all jobs are listed: {count}"

    def test_abort_job_self(self):
        """Test TRAIN_SELF, users can only abort jobs on their own sites"""

        job_a = MockAdminServer.job_def_manager.find_job_by_name("job_a")
        response = TestAdminAuth.admin_server_a.abort_job(job_a.job_id)
        assert self._find_type(response, "success")

        job_b = MockAdminServer.job_def_manager.find_job_by_name("job_b")
        response = TestAdminAuth.admin_server_b.abort_job(job_b.job_id)
        assert self._find_type(response, "success")

    def test_abort_job_other(self):
        """Test negative TRAIN_SELF, users can not abort jobs on other's sites"""

        job_b = MockAdminServer.job_def_manager.find_job_by_name("job_b")
        response = TestAdminAuth.admin_server_a.abort_job(job_b.job_id)
        assert AUTH_ERROR in self._get_type(response, "error")

        job_a = MockAdminServer.job_def_manager.find_job_by_name("job_a")
        response = TestAdminAuth.admin_server_b.abort_job(job_a.job_id)
        assert AUTH_ERROR in self._get_type(response, "error")

    def test_abort_job_super(self):
        """Test TRAIN_ALL, superusers can abort jobs on all sites"""

        job_a = MockAdminServer.job_def_manager.find_job_by_name("job_a")
        response = TestAdminAuth.admin_server_super.abort_job(job_a.job_id)
        assert self._find_type(response, "success")

        job_b = MockAdminServer.job_def_manager.find_job_by_name("job_b")
        response = TestAdminAuth.admin_server_super.abort_job(job_b.job_id)
        assert self._find_type(response, "success")

    @staticmethod
    def _find_type(result: dict, data_type: str):
        for item in result["data"]:
            if item["type"] == data_type:
                return True

        return False

    @staticmethod
    def _get_type(result: dict, data_type: str) -> str:
        for item in result["data"]:
            if item["type"] == data_type:
                return item["data"]

        return ""
