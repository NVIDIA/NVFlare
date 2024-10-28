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

import json
import os
import shutil

from nvflare.apis.app_deployer_spec import AppDeployerSpec, FLContext
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import SystemComponents, SystemVarName
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.utils.job_utils import load_job_def_bytes
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.dict_utils import update_components


class HubAppDeployer(AppDeployerSpec, FLComponent):

    HUB_CLIENT_CONFIG_TEMPLATE_NAME = "hub_client.json"
    OLD_HUB_CLIENT_CONFIG_TEMPLATE_NAME = "t1_config_fed_client.json"
    HUB_SERVER_CONFIG_TEMPLATE_NAME = "hub_server.json"
    OLD_HUB_SERVER_CONFIG_TEMPLATE_NAME = "t2_server_components.json"

    HUB_CLIENT_CONFIG_TEMPLATES = [HUB_CLIENT_CONFIG_TEMPLATE_NAME, OLD_HUB_CLIENT_CONFIG_TEMPLATE_NAME]
    HUB_SERVER_CONFIG_TEMPLATES = [HUB_SERVER_CONFIG_TEMPLATE_NAME, OLD_HUB_SERVER_CONFIG_TEMPLATE_NAME]

    def __init__(self):
        FLComponent.__init__(self)

    def prepare(
        self, fl_ctx: FLContext, workspace: Workspace, job_id: str, remove_tmp_t2_dir: bool = True
    ) -> (str, dict, bytes):
        """
        Prepare T2 job

        Args:
            fl_ctx:
            workspace:
            job_id:
            remove_tmp_t2_dir:

        Returns: error str if any, meta dict, and job bytes to be submitted to T2 store

        """
        t1_workspace_dir = fl_ctx.get_prop(SystemVarName.WORKSPACE)

        fed_client = fl_ctx.get_prop(SystemComponents.FED_CLIENT)
        cell = fed_client.cell
        t1_root_url = cell.get_root_url_for_child()
        t1_secure_train = cell.is_secure()

        server_app_config_path = workspace.get_server_app_config_file_path(job_id)
        if not os.path.exists(server_app_config_path):
            return f"missing {server_app_config_path}", None, None

        # step 2: make a copy of the app for T2
        t1_run_dir = workspace.get_run_dir(job_id)
        t2_job_id = job_id + "_t2"  # temporary ID for creating T2 job
        t2_run_dir = workspace.get_run_dir(t2_job_id)
        shutil.copytree(t1_run_dir, t2_run_dir)

        # step 3: modify the T1 client's config_fed_client.json to use HubExecutor
        # simply use t1_config_fed_client.json in the site folder
        site_config_dir = workspace.get_site_config_dir()
        t1_client_app_config_path = workspace.get_file_path_in_site_config(self.HUB_CLIENT_CONFIG_TEMPLATES)

        if not t1_client_app_config_path:
            return (
                f"no HUB client config template '{self.HUB_CLIENT_CONFIG_TEMPLATES}' in {site_config_dir}",
                None,
                None,
            )

        shutil.copyfile(t1_client_app_config_path, workspace.get_client_app_config_file_path(job_id))

        # step 4: modify T2 server's config_fed_server.json to use HubController
        t2_server_app_config_path = workspace.get_server_app_config_file_path(t2_job_id)
        if not os.path.exists(t2_server_app_config_path):
            return f"missing {t2_server_app_config_path}", None, None

        t2_server_component_file = workspace.get_file_path_in_site_config(self.HUB_SERVER_CONFIG_TEMPLATES)

        if not t2_server_component_file:
            return (
                f"no HUB server config template '{self.HUB_SERVER_CONFIG_TEMPLATES}' in {site_config_dir}",
                None,
                None,
            )

        with open(t2_server_app_config_path) as file:
            t2_server_app_config_dict = json.load(file)

        with open(t2_server_component_file) as file:
            t2_server_component_dict = json.load(file)

        # update components in the server's config with changed components
        # This will replace shareable_generator with the one defined in t2_server_components.json
        err = update_components(target_dict=t2_server_app_config_dict, from_dict=t2_server_component_dict)
        if err:
            return err

        # change to use HubController as the workflow for T2
        t2_wf = t2_server_component_dict.get("workflows", None)
        if not t2_wf:
            return f"missing workflows in {t2_server_component_file}", None, None
        t2_server_app_config_dict["workflows"] = t2_wf

        # add T1's env vars to T2 server app config so that they can be used in config definition
        t2_server_app_config_dict.update(
            {
                "T1_WORKSPACE": t1_workspace_dir,
                "T1_ROOT_URL": t1_root_url,
                "T1_SECURE_TRAIN": t1_secure_train,
            }
        )

        # recreate T2's server app config file
        with open(t2_server_app_config_path, "w") as f:
            json.dump(t2_server_app_config_dict, f, indent=4)

        # create job meta for T2
        t1_meta_path = workspace.get_job_meta_path(job_id)
        if not os.path.exists(t1_meta_path):
            return f"missing {t1_meta_path}", None, None
        with open(t1_meta_path) as file:
            t1_meta = json.load(file)

        submitter_name = t1_meta.get(JobMetaKey.SUBMITTER_NAME.value, "")
        submitter_org = t1_meta.get(JobMetaKey.SUBMITTER_ORG.value, "")
        submitter_role = t1_meta.get(JobMetaKey.SUBMITTER_ROLE.value, "")
        scope = t1_meta.get(JobMetaKey.SCOPE.value, "")

        # Note: the app_name is already created like "app_"+site_name, which is also the directory that contains
        # app config files (config_fed_server.json and config_fed_client.json).
        # We need to make sure that the deploy-map uses this app name!
        # We also add the FROM_HUB_SITE into the T2's job meta to indicate that this job comes from a HUB site.
        t2_app_name = "app_" + workspace.site_name
        t2_meta = {
            "name": t2_app_name,
            "deploy_map": {t2_app_name: ["@ALL"]},
            "min_clients": 1,
            "job_id": job_id,
            JobMetaKey.SUBMITTER_NAME.value: submitter_name,
            JobMetaKey.SUBMITTER_ORG.value: submitter_org,
            JobMetaKey.SUBMITTER_ROLE.value: submitter_role,
            JobMetaKey.SCOPE.value: scope,
            JobMetaKey.FROM_HUB_SITE.value: workspace.site_name,
        }

        t2_meta_path = workspace.get_job_meta_path(t2_job_id)
        with open(t2_meta_path, "w") as f:
            json.dump(t2_meta, f, indent=4)

        # step 5: submit T2 app (as a job) to T1's job store
        t2_job_def = load_job_def_bytes(from_path=workspace.root_dir, def_name=t2_job_id)

        job_validator = fl_ctx.get_prop(SystemComponents.JOB_META_VALIDATOR)
        valid, error, meta = job_validator.validate(t2_job_id, t2_job_def)
        if not valid:
            return f"invalid T2 job definition: {error}", None, None

        # make sure meta contains the right job ID
        t2_jid = meta.get(JobMetaKey.JOB_ID.value, None)
        if not t2_jid:
            return "missing Job ID from T2 meta!", None, None

        if job_id != t2_jid:
            return f"T2 Job ID {t2_jid} != T1 Job ID {job_id}", None, None

        # step 6: remove the temporary job def for T2
        if remove_tmp_t2_dir:
            shutil.rmtree(t2_run_dir)
        return "", meta, t2_job_def

    def deploy(
        self, workspace: Workspace, job_id: str, job_meta: dict, app_name: str, app_data: bytes, fl_ctx: FLContext
    ) -> str:
        # step 1: deploy the T1 app into the workspace
        deployer = fl_ctx.get_prop(SystemComponents.DEFAULT_APP_DEPLOYER)
        err = deployer.deploy(workspace, job_id, job_meta, app_name, app_data, fl_ctx)
        if err:
            self.log_error(fl_ctx, f"Failed to deploy job {job_id}: {err}")
            return err

        err, meta, t2_job_def = self.prepare(fl_ctx, workspace, job_id)
        if err:
            self.log_error(fl_ctx, f"Failed to deploy job {job_id}: {err}")
            return err

        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        if not isinstance(job_manager, JobDefManagerSpec):
            return "Job Manager for T2 not configured!"
        job_manager.create(meta, t2_job_def, fl_ctx)
        return ""
