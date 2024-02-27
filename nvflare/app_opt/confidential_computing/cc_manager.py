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
from typing import Dict

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_opt.confidential_computing.cc_authorizer import TokenPundit

# from .cc_helper import CCHelper

PEER_CTX_CC_TOKEN = "_peer_ctx_cc_token"
CC_TOKEN = "_cc_token"
CC_NAMESPACE = "_cc_namespace"
CC_INFO = "_cc_info"
CC_TOKEN_VALIDATED = "_cc_token_validated"


class CCManager(FLComponent):
    def __init__(self, cc_issuer_id: str, cc_verifier_ids: [str]):
        """Manage all confidential computing related tasks.

        This manager does the following tasks:
        obtaining its own GPU CC token
        preparing the token to the server
        keeping clients' tokens in server
        validating all tokens in the entire NVFlare system

        Args:

        """
        FLComponent.__init__(self)
        self.site_name = None
        self.cc_issuer_id = cc_issuer_id
        self.cc_verifier_ids = cc_verifier_ids
        self.cc_issuer = None
        self.cc_verifiers = {}
        self.my_token = None
        self.participant_cc_info = {}  # used by the Server to keep tokens of all clients

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_BOOTSTRAP:
            try:
                self._setup_cc_authorizers(fl_ctx)

                err = self._prepare_for_attestation(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in attestation preparation")
                err = "exception in attestation preparation"
            finally:
                if err:
                    self.log_critical(fl_ctx, err, fire_event=False)
                    raise UnsafeComponentError(err)
        elif event_type == EventType.BEFORE_CLIENT_REGISTER:
            # On client side
            self._prepare_token_for_login(fl_ctx)
        elif event_type == EventType.CLIENT_REGISTERED:
            # Server side
            self._add_client_token(fl_ctx)
        elif event_type == EventType.CLIENT_QUIT:
            # Server side
            self._remove_client_token(fl_ctx)
        elif event_type == EventType.AUTHORIZE_COMMAND_CHECK:
            command_to_check = fl_ctx.get_prop(key=FLContextKey.COMMAND_NAME)
            self.logger.debug(f"Received {command_to_check=}")
            if command_to_check == AdminCommandNames.CHECK_RESOURCES:
                try:
                    err = self._client_to_check_participant_token(fl_ctx)
                except:
                    self.log_exception(fl_ctx, "exception in validating participants")
                    err = "Participants unable to meet client CC requirements"
                finally:
                    if err:
                        self._not_authorize_job(err, fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            # Server side
            try:
                err = self._server_to_check_client_token(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in validating clients")
                err = "Clients unable to meet server CC requirements"
            finally:
                if err:
                    self._block_job(err, fl_ctx)
        elif event_type == EventType.AFTER_CHECK_CLIENT_RESOURCES:
            # Server side
            fl_ctx.remove_prop(PEER_CTX_CC_TOKEN)

    def _setup_cc_authorizers(self, fl_ctx):
        engine = fl_ctx.get_engine()
        self.cc_issuer = engine.get_component(self.cc_issuer_id)
        if not isinstance(self.cc_issuer, TokenPundit):
            raise RuntimeError(f"cc_authorizer_id {self.cc_issuer_id} must be a TokenPundit, but got {self.cc_issuer.__class__}")

        for id in self.cc_verifier_ids:
            authorizer = engine.get_component(id)
            if not isinstance(authorizer, TokenPundit):
                raise RuntimeError(f"cc_authorizer_id {id} must be a TokenPundit, but got {authorizer.__class__}")
            self.cc_verifiers[authorizer.get_namespace()] = authorizer

    def _prepare_token_for_login(self, fl_ctx: FLContext):
        # client side
        if self.my_token is None:
            self.my_token = self.cc_issuer.generate()
        cc_info = {CC_TOKEN: self.my_token, CC_NAMESPACE: self.cc_issuer.get_namespace()}
        fl_ctx.set_prop(key=CC_INFO, value=cc_info, sticky=False, private=False)

    def _add_client_token(self, fl_ctx: FLContext):
        # server side
        peer_ctx = fl_ctx.get_peer_context()
        token_owner = peer_ctx.get_identity_name()
        peer_cc_info = peer_ctx.get_prop(CC_INFO, {CC_TOKEN: "", CC_NAMESPACE: ""})
        self.participant_cc_info[token_owner] = peer_cc_info
        self.participant_cc_info[token_owner][CC_TOKEN_VALIDATED] = False

        self.logger.info(f"Added CC client: {token_owner} token: {peer_cc_info[CC_TOKEN]}")

    def _remove_client_token(self, fl_ctx: FLContext):
        # server side
        peer_ctx = fl_ctx.get_peer_context()
        token_owner = peer_ctx.get_identity_name()
        self.participant_cc_info.pop(token_owner)
        self.logger.info(f"Removed CC client: {token_owner}")

    def _prepare_for_attestation(self, fl_ctx: FLContext) -> str:
        # both server and client sides
        self.site_name = fl_ctx.get_identity_name()
        workspace_folder = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT).get_site_config_dir()
        # self.helper = CCHelper(site_name=self.site_name, verifiers=self.verifiers)
        # self.helper = TDXCCHelper(site_name=self.site_name,
        #                           tdx_cli_command="/home/azureuser/TDX/client/tdx-cli/trustauthority-cli",
        #                           config_dir=workspace_folder)
        # ok = self.helper.prepare()
        # if not ok:
        #     return "failed to attest"

        # self.cc_authorizer = TDXConnector(tdx_cli_command="/home/azureuser/TDX/client/tdx-cli/trustauthority-cli",
        #                              config_dir=workspace_folder)

        # self.cc_authorizer = self._get_authorizer()
        self.my_token = self.cc_issuer.generate()
        if not self.my_token:
            return "failed to get CC token"

        self.logger.info(f"site: {self.site_name} got the token: {self.my_token}")
        self.participant_cc_info[self.site_name] = {CC_TOKEN: self.my_token, CC_NAMESPACE: self.cc_issuer.get_namespace(), CC_TOKEN_VALIDATED: True}
        return ""

    def _client_to_check_participant_token(self, fl_ctx: FLContext) -> str:
        # Client side
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx is None:
            return f"Empty peer context in {self.site_name=}"
        participants_to_validate = peer_ctx.get_prop(PEER_CTX_CC_TOKEN, None)
        if not participants_to_validate:
            return "missing PEER_CTX_CC_TOKEN prop in peer context"

        if not isinstance(participants_to_validate, dict):
            return (
                f"bad PEER_CTX_CC_TOKEN prop in peer context: must be a dict but got {type(participants_to_validate)}"
            )

        if not participants_to_validate:
            return ""

        return self._validate_participants_tokens(participants_to_validate)

    def _server_to_check_client_token(self, fl_ctx: FLContext) -> str:
        participants = fl_ctx.get_prop(FLContextKey.JOB_PARTICIPANTS)
        if not participants:
            return f"missing '{FLContextKey.JOB_PARTICIPANTS}' prop in fl_ctx"

        if not isinstance(participants, list):
            return f"bad value for {FLContextKey.JOB_PARTICIPANTS} in fl_ctx: expect list bot got {type(participants)}"

        # participant_tokens = {self.site_name: self.my_token, CC_VERIFIER: self.cc_issuer}
        participant_tokens = {self.site_name: self.participant_cc_info[self.site_name]}
        for p in participants:
            assert isinstance(p, str)
            if p == self.site_name:
                continue
            if p not in self.participant_cc_info:
                return f"no token available for participant {p}"
            if self.participant_cc_info.get(p):
                # participant_tokens[p] = self.participant_cc_info[p][CC_TOKEN]
                participant_tokens[p] = self.participant_cc_info[p]
            else:
                participant_tokens[p] = {}

        err = self._validate_participants_tokens(participant_tokens)
        if err:
            return err

        for p in participant_tokens:
            self.participant_cc_info[p][CC_TOKEN_VALIDATED] = True
        fl_ctx.set_prop(key=PEER_CTX_CC_TOKEN, value=participant_tokens, sticky=True, private=False)
        self.logger.info(f"{self.site_name=} set PEER_CTX_CC_TOKEN with {participant_tokens=}")
        return ""

    def _validate_participants_tokens(self, participants) -> str:
        self.logger.debug(f"Validating participant tokens {participants=}")
        result = self._validate_participants(participants)
        assert isinstance(result, dict)
        for p in result:
            self.participant_cc_info[p] = {CC_TOKEN: participants[p], CC_TOKEN_VALIDATED: True}
        invalid_participant_list = [k for k, v in self.participant_cc_info.items() if v[CC_TOKEN_VALIDATED] is False]
        if invalid_participant_list:
            invalid_participant_string = ",".join(invalid_participant_list)
            self.logger.debug(f"{invalid_participant_list=}")
            return f"Participant {invalid_participant_string} not meeting CC requirements"
        else:
            return ""

    def _validate_participants(self, participants: Dict[str, {}]) -> Dict[str, bool]:
        result = {}
        if not participants:
            return result
        for k, v in participants.items():
            token = v.get(CC_TOKEN, "")
            verifier = self.cc_verifiers.get(v.get(CC_NAMESPACE, ""), None)
            if verifier:
                if verifier.verify(token):
                    result[k] = True
        self.logger.info(f"CC - results from validating participants' tokens: {result}")
        return result

    def _not_authorize_job(self, reason: str, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID, "")
        self.log_error(fl_ctx, f"Job {job_id} is blocked: {reason}")
        fl_ctx.set_prop(key=FLContextKey.AUTHORIZATION_REASON, value=reason, sticky=False)
        fl_ctx.set_prop(key=FLContextKey.AUTHORIZATION_RESULT, value=False, sticky=False)

    def _block_job(self, reason: str, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID, "")
        self.log_error(fl_ctx, f"Job {job_id} is blocked: {reason}")
        fl_ctx.set_prop(key=FLContextKey.JOB_BLOCK_REASON, value=reason, sticky=False)
        fl_ctx.set_prop(key=FLContextKey.AUTHORIZATION_RESULT, value=False, sticky=False)
