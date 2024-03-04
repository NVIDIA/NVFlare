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
import threading
from typing import Dict, List

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, RunProcessKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

PEER_CTX_CC_TOKEN = "_peer_ctx_cc_token"
CC_TOKEN = "_cc_token"
CC_ISSUER = "_cc_issuer"
CC_NAMESPACE = "_cc_namespace"
CC_INFO = "_cc_info"
CC_TOKEN_VALIDATED = "_cc_token_validated"


class CCManager(FLComponent):
    def __init__(self, cc_issuer_ids: [str], cc_verifier_ids: [str]):
        """Manage all confidential computing related tasks.

        This manager does the following tasks:
        obtaining its own CC token
        preparing the token to the server
        keeping clients' tokens in server
        validating all tokens in the entire NVFlare system

        Args:

        """
        FLComponent.__init__(self)
        self.site_name = None
        self.cc_issuer_ids = cc_issuer_ids
        self.cc_verifier_ids = cc_verifier_ids
        self.cc_issuers = []
        self.cc_verifiers = {}
        self.participant_cc_info = {}  # used by the Server to keep tokens of all clients

        self.lock = threading.Lock()

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
        elif event_type == EventType.BEFORE_CLIENT_REGISTER or event_type == EventType.BEFORE_CLIENT_HEARTBEAT:
            # On client side
            self._prepare_token_for_login(fl_ctx)
        elif event_type == EventType.CLIENT_REGISTERED or event_type == EventType.AFTER_CLIENT_HEARTBEAT:
            # Server side
            self._add_client_token(fl_ctx)
        elif event_type == EventType.CLIENT_QUIT:
            # Server side
            self._remove_client_token(fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
            # Client side: check resources before job scheduled
            try:
                err = self._client_to_check_participant_token(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in validating participants")
                err = "Participants unable to meet client CC requirements"
            finally:
                if err:
                    self._not_authorize_job(err, fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            # Server side: job scheduler check client resources
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
        for i_id in self.cc_issuer_ids:
            issuer = engine.get_component(i_id)
            if not isinstance(issuer, CCAuthorizer):
                raise RuntimeError(f"cc_issuer_id {i_id} must be a CCAuthorizer, but got {issuer.__class__}")
            self.cc_issuers.append(issuer)

        for v_id in self.cc_verifier_ids:
            authorizer = engine.get_component(v_id)
            if not isinstance(authorizer, CCAuthorizer):
                raise RuntimeError(f"cc_authorizer_id {v_id} must be a CCAuthorizer, but got {authorizer.__class__}")
            namespace = authorizer.get_namespace()
            if namespace in self.cc_verifiers.keys():
                raise RuntimeError(f"Authorizer with namespace: {namespace} already exist.")
            self.cc_verifiers[namespace] = authorizer

    def _prepare_token_for_login(self, fl_ctx: FLContext):
        # client side: if token expired then generate a new one
        self._handle_expired_tokens()

        site_cc_info = self.participant_cc_info[self.site_name]
        cc_info = self._get_participant_tokens(site_cc_info)
        fl_ctx.set_prop(key=CC_INFO, value=cc_info, sticky=False, private=False)

    def _add_client_token(self, fl_ctx: FLContext):
        # server side
        peer_ctx = fl_ctx.get_peer_context()
        token_owner = peer_ctx.get_identity_name()
        peer_cc_info = peer_ctx.get_prop(CC_INFO, [{CC_TOKEN: "", CC_NAMESPACE: ""}])
        new_tokens = []
        for i in peer_cc_info:
            new_tokens.append(i[CC_TOKEN])

        old_cc_info = self.participant_cc_info.get(token_owner)
        old_tokens = []
        if old_cc_info:
            for i in old_cc_info:
                old_tokens.append(i[CC_TOKEN])

        if not old_cc_info or set(new_tokens) != set(old_tokens):
            self.participant_cc_info[token_owner] = peer_cc_info
            self.logger.info(f"Added CC client: {token_owner} tokens: {peer_cc_info}")

            with self.lock:
                self._verify_running_jobs(fl_ctx)

    def _verify_running_jobs(self, fl_ctx):
        engine = fl_ctx.get_engine()
        run_processes = engine.run_processes
        running_jobs = list(run_processes.keys())
        for job_id in running_jobs:
            job_participants = run_processes[job_id].get(RunProcessKey.PARTICIPANTS)
            participants = []
            for _, client in job_participants.items():
                participants.append(client.name)

            participant_tokens = {}
            err = self._verify_participants(participants, participant_tokens)
            if err:
                engine.job_runner.stop_run(job_id, fl_ctx)
                self.logger.info(f"Stop Job: {job_id} with CC verification error: {err} ")

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

        self.participant_cc_info[self.site_name] = []
        for issuer in self.cc_issuers:
            my_token = issuer.generate()
            namespace = issuer.get_namespace()

            if not my_token:
                return "failed to get CC token"

            self.logger.info(f"site: {self.site_name} namespace: {namespace} got the token: {my_token}")
            cc_info = {CC_TOKEN: my_token, CC_ISSUER: issuer, CC_NAMESPACE: namespace, CC_TOKEN_VALIDATED: True}
            self.participant_cc_info[self.site_name].append(cc_info)

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

        participant_tokens = {}
        err = self._verify_participants(participants, participant_tokens)
        if err:
            return err

        fl_ctx.set_prop(key=PEER_CTX_CC_TOKEN, value=participant_tokens, sticky=True, private=False)
        self.logger.info(f"{self.site_name=} set PEER_CTX_CC_TOKEN with {participant_tokens=}")
        return ""

    def _verify_participants(self, participants, participant_tokens):
        # if server token expired, then generates a new one
        self._handle_expired_tokens()

        site_cc_info = self.participant_cc_info[self.site_name]
        participant_tokens[self.site_name] = self._get_participant_tokens(site_cc_info)

        for p in participants:
            assert isinstance(p, str)
            if p == self.site_name:
                continue
            if p not in self.participant_cc_info:
                return f"no token available for participant {p}"
            if self.participant_cc_info.get(p):
                participant_tokens[p] = self._get_participant_tokens(self.participant_cc_info[p])
            else:
                participant_tokens[p] = [{}]
        return self._validate_participants_tokens(participant_tokens)

    def _get_participant_tokens(self, site_cc_info):
        cc_info = []
        for i in site_cc_info:
            namespace = i.get(CC_NAMESPACE)
            token = i.get(CC_TOKEN)
            cc_info.append({CC_TOKEN: token, CC_NAMESPACE: namespace, CC_TOKEN_VALIDATED: False})
        return cc_info

    def _handle_expired_tokens(self):
        site_cc_info = self.participant_cc_info[self.site_name]
        for i in site_cc_info:
            issuer = i.get(CC_ISSUER)
            token = i.get(CC_TOKEN)
            if not issuer.verify(token):
                token = issuer.generate()
                if not token:
                    raise RuntimeError(f"{self.site_name} failed to generate a new CC token")
                i[CC_TOKEN] = token
                self.logger.info(f"site: {self.site_name} namespace: {issuer.get_namespace()} got a new CC token: {token}")

    def _validate_participants_tokens(self, participants) -> str:
        self.logger.debug(f"Validating participant tokens {participants=}")
        result, invalid_participant_list = self._validate_participants(participants)
        if invalid_participant_list:
            invalid_participant_string = ",".join(invalid_participant_list)
            self.logger.debug(f"{invalid_participant_list=}")
            return f"Participant {invalid_participant_string} not meeting CC requirements"
        else:
            return ""

    def _validate_participants(self, participants: Dict[str, List[Dict[str, str]]]) -> (Dict[str, bool], List[str]):
        result = {}
        invalid_participant_list = []
        if not participants:
            return result

        for k, cc_info in participants.items():
            for v in cc_info:
                token = v.get(CC_TOKEN, "")
                namespace = v.get(CC_NAMESPACE, "")
                verifier = self.cc_verifiers.get(namespace, None)
                if verifier and verifier.verify(token):
                    result[k + "." + namespace] = True
                else:
                    invalid_participant_list.append(k + " namespace: {" + namespace + "}")
        self.logger.info(f"CC - results from validating participants' tokens: {result}")
        return result, invalid_participant_list

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
