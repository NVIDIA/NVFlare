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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.fuel.sec.security_content_service import SecurityContentService, LoadResult

from .attestation_helper import AttestationHelper


_KEY_PARTICIPANTS = "cc.participants"
_DEFAULT_CONFIG_FILE_NAME = "config_cc.json"
_CLAIM_POLICY_FILE_NAME = "cc_claim.json"
_REQUIREMENT_POLICY_FILE_NAME = "cc_requirement.json"


class AttestationManager(FLComponent):

    def __init__(self,
                 config_file_name: str=_DEFAULT_CONFIG_FILE_NAME,
                 claim_policy_file_name: str=_CLAIM_POLICY_FILE_NAME,
                 requirement_policy_file_name: str=_REQUIREMENT_POLICY_FILE_NAME):
        FLComponent.__init__(self)
        self.helper = None
        self.config_file_name = config_file_name
        self.claim_policy_file_name = claim_policy_file_name
        self.requirement_policy_file_name = requirement_policy_file_name

    def _prepare_for_attestation(self, fl_ctx: FLContext) -> str:
        engine = fl_ctx.get_engine()
        workspace: Workspace = engine.get_workspace()

        claim_files = [self.claim_policy_file_name, self.claim_policy_file_name + '.default']
        claim_path = workspace.get_file_path_in_site_config(claim_files)
        if not claim_path:
            return f"missing claim policy file {claim_files}"

        req_files = [self.requirement_policy_file_name, self.requirement_policy_file_name + '.default']
        req_path = workspace.get_file_path_in_site_config(req_files)
        if not req_path:
            return f"missing requirement policy file {req_files}"

        config_data, rc = SecurityContentService.load_json(self.config_file_name)
        if rc != LoadResult.OK:
            return f"failed to load CC config file '{self.config_file_name}': {rc}"

        if not isinstance(config_data, dict):
            return f"invalid CC config file '{self.config_file_name}': data must be dict but got {type(config_data)}"

        attestation_service_endpoint = config_data.get("attestation_service_endpoint", None)
        if not attestation_service_endpoint:
            return f"invalid CC config file '{self.config_file_name}': missing attestation_service_endpoint"

        orchestration_server_endpoint = config_data.get("orchestration_server_endpoint", None)
        if not orchestration_server_endpoint:
            return f"invalid CC config file '{self.config_file_name}': missing orchestration_server_endpoint"

        self.helper = AttestationHelper(
            site_name=fl_ctx.get_identity_name(),
            attestation_service_endpoint=attestation_service_endpoint,
            orchestration_server_endpoint=orchestration_server_endpoint
        )

        return self.helper.prepare(
            claim_policy_file_path=claim_path,
            requirement_policy_file_path=req_path
        )

    def _block_job(self, reason: str, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID, "")
        self.log_error(fl_ctx, f"Job {job_id} is blocked: {reason}")
        fl_ctx.set_prop(
            key=FLContextKey.JOB_BLOCK_REASON,
            value=reason
        )

    def _check_participants_for_client(self, fl_ctx: FLContext) -> str:
        resource_specs = fl_ctx.get_prop(FLContextKey.CLIENT_RESOURCE_SPECS, None)
        if not resource_specs:
            return f"missing '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx"

        if not isinstance(resource_specs, dict):
            return f"bad '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx: "\
                f"must be a dict but got {type(resource_specs)}"

        participants = resource_specs.get(_KEY_PARTICIPANTS, None)
        if not participants:
            return f"bad '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx: "\
                f"missing '{_KEY_PARTICIPANTS}'"

        return self.helper.validate_participants(participants)

    def _check_participants_for_server(self, fl_ctx: FLContext) -> str:
        participants = fl_ctx.get_prop(FLContextKey.JOB_PARTICIPANTS)
        if not participants:
            return f"missing '{FLContextKey.JOB_PARTICIPANTS}' prop in fl_ctx"

        err = self.helper.validate_participants(participants)
        if err:
            return err

        resource_specs = fl_ctx.get_prop(FLContextKey.CLIENT_RESOURCE_SPECS, None)
        if resource_specs is None:
            return f"missing '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx"

        # add "participants" to each client's resource spec so the client side can validate
        for client_name, spec in resource_specs:
            if not isinstance(spec, dict):
                return f"bad resource spec for client {client_name}: expect dict but got {type(spec)}"
            spec[_KEY_PARTICIPANTS] = participants
        return ""

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            try:
                err = self._prepare_for_attestation(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in attestation preparation")
                err = "exception occurred"
            if err:
                self.system_panic(
                    reason=f"Failed to prepare for attestation: {err}",
                    fl_ctx=fl_ctx
                )
        elif event_type == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
            # this happens on the Client side
            try:
                err = self._check_participants_for_client(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in validating participants")
                err = "exception occurred"
            if err:
                self._block_job(err, fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            # this happens on the Server side
            try:
                err = self._check_participants_for_server(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in validating participants")
                err = "exception occurred"
            if err:
                self._block_job(err, fl_ctx)
