# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import os
import shutil

from nvflare.apis.app_deployer_spec import AppDeployerSpec
from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.zip_utils import unzip_all_from_bytes
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE
from nvflare.lighter.utils import verify_folder_signature_and_get_signers
from nvflare.private.privacy_manager import PrivacyService
from nvflare.security.logging import secure_format_exception

from .app_authz import AppAuthzService


class AppDeployer(AppDeployerSpec):
    def deploy(
        self, workspace: Workspace, job_id: str, job_meta: dict, app_name: str, app_data: bytes, fl_ctx: FLContext
    ) -> str:
        """Deploys the app.

        Returns:
            error message if any
        """
        privacy_scope = job_meta.get(JobMetaKey.SCOPE, "")

        # check whether this scope is allowed
        if not PrivacyService.is_scope_allowed(privacy_scope):
            return f"privacy scope '{privacy_scope}' is not allowed"

        try:
            run_dir = workspace.get_run_dir(job_id)
            app_path = workspace.get_app_dir(job_id)
            app_file = os.path.join(run_dir, "fl_app.txt")
            job_meta_file = workspace.get_job_meta_path(job_id)

            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            if not os.path.exists(app_path):
                os.makedirs(app_path)

            unzip_all_from_bytes(app_data, app_path)

            with open(app_file, "wt") as f:
                f.write(f"{app_name}")

            submitter_name = job_meta.get(JobMetaKey.SUBMITTER_NAME, "")
            submitter_org = job_meta.get(JobMetaKey.SUBMITTER_ORG, "")
            submitter_role = job_meta.get(JobMetaKey.SUBMITTER_ROLE, "")
            authz_submitters = [(submitter_name, submitter_org, submitter_role)]

            sig_file = os.path.join(app_path, NVFLARE_SIG_FILE)
            if os.path.exists(sig_file):
                root_ca_path = workspace.get_file_path_in_startup("rootCA.pem")
                verified, signers = verify_folder_signature_and_get_signers(app_path, root_ca_path)
                if not verified:
                    if os.path.exists(run_dir):
                        shutil.rmtree(run_dir, ignore_errors=True)
                    return f"app {app_name} does not pass signature verification"
                if not signers:
                    if os.path.exists(run_dir):
                        shutil.rmtree(run_dir, ignore_errors=True)
                    return f"app {app_name}: signature verified but no signer identity could be extracted"
                authz_submitters = signers

            err, app_info = AppAuthzService.validate_app(app_path)
            if err:
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir, ignore_errors=True)
                return err

            job_meta = copy.deepcopy(job_meta)
            if app_info.get(AppValidationKey.BYOC, False):
                job_meta[AppValidationKey.BYOC] = True
            else:
                job_meta.pop(AppValidationKey.BYOC, None)
            with open(job_meta_file, "w") as f:
                json.dump(job_meta, f, indent=4)

            for authz_name, authz_org, authz_role in authz_submitters:
                authorized, err = AppAuthzService.authorize_app_info(
                    app_info=app_info,
                    submitter_name=authz_name,
                    submitter_org=authz_org,
                    submitter_role=authz_role,
                    job_meta=job_meta,
                )
                if err or not authorized:
                    if os.path.exists(run_dir):
                        shutil.rmtree(run_dir, ignore_errors=True)
                    return err if err else "not authorized"

        except Exception as e:
            raise Exception(f"exception {secure_format_exception(e)} when deploying app {app_name}")
