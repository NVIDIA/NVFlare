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
import time
from typing import Dict, List, Tuple

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, RunProcessKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer, CCTokenGenerateError, CCTokenVerifyError
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server.training_cmds import TrainingCommandModule

PEER_CTX_CC_TOKEN = "_peer_ctx_cc_token"
CC_TOKEN = "_cc_token"
CC_ISSUER = "_cc_issuer"
CC_NAMESPACE = "_cc_namespace"
CC_INFO = "_cc_info"
CC_TOKEN_VALIDATED = "_cc_token_validated"
CC_VERIFY_ERROR = "_cc_verify_error."

CC_ISSUER_ID = "issuer_id"
TOKEN_GENERATION_TIME = "token_generation_time"
TOKEN_EXPIRATION = "token_expiration"

SHUTDOWN_SYSTEM = 1
SHUTDOWN_JOB = 2

CC_VERIFICATION_FAILED = "not meeting CC requirements"


class CCManager(FLComponent):
    def __init__(
        self,
        cc_issuers_conf: List[Dict[str, str]],
        cc_verifier_ids: List[str],
        verify_frequency: int = 600,
        critical_level=SHUTDOWN_JOB,
        cc_enabled_sites: List[str] = [],
    ):
        """Manage all confidential computing related tasks.

        This manager does the following tasks:
            1. obtaining its own CC token
            2. preparing the token to the server
            3. keeping clients' tokens in server
            4. validating all tokens in the entire NVFlare system
            5. not allowing the system to start if failed to get CC token
            6. shutdown the running jobs if CC tokens expired

        # TODO: should we separate the server and client side into two components?

        Args:
            cc_issuers_conf: configuration of the CC token issuers. each contains the CC token issuer component ID,
                            and the token expiration time
            cc_verifier_ids: CC token verifiers component IDs
            verify_frequency: CC tokens verification frequency
            critical_level: critical level for shutting down the system or jobs
            cc_enabled_sites: list of sites that are enabled for CC
        """
        FLComponent.__init__(self)
        self.site_name = None
        self.cc_issuers_conf = cc_issuers_conf
        self.cc_verifier_ids = cc_verifier_ids
        self.cc_enabled_sites = cc_enabled_sites

        if not isinstance(verify_frequency, int):
            raise ValueError(f"verify_frequency must be int, but got {type(verify_frequency).__name__}")
        self.verify_frequency = int(verify_frequency)

        self.critical_level = critical_level
        if self.critical_level not in [SHUTDOWN_SYSTEM, SHUTDOWN_JOB]:
            raise ValueError(f"critical_level must be in [{SHUTDOWN_SYSTEM}, {SHUTDOWN_JOB}]. But got {critical_level}")

        self.verify_time = None
        self.cc_issuers = {}
        self.cc_verifiers = {}
        self.participant_cc_info = {}  # used by the Server to keep tokens of all clients

        self.token_submitted = False
        self.lock = threading.Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_BOOTSTRAP:
            err = None
            try:
                self._setup_cc_authorizers(fl_ctx)

                err = self._generate_tokens(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in attestation preparation")
                err = "exception in attestation preparation"
            finally:
                if err:
                    self.log_critical(fl_ctx, err, fire_event=False)
                    raise UnsafeComponentError(err)
        elif event_type == EventType.BEFORE_CLIENT_REGISTER or event_type == EventType.BEFORE_CLIENT_HEARTBEAT:
            # On client side
            self._prepare_cc_info(fl_ctx)
        elif event_type == EventType.CLIENT_REGISTER_RECEIVED or event_type == EventType.CLIENT_HEARTBEAT_RECEIVED:
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
                    self._block_job(err, fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            # Server side: job scheduler check client resources
            try:
                err = self._server_to_check_client_token(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in validating clients")
                err = "Clients unable to meet server CC requirements"
            finally:
                if err:
                    if self.critical_level == SHUTDOWN_JOB:
                        self._block_job(err, fl_ctx)
                    else:
                        threading.Thread(target=self._shutdown_system, args=[err, fl_ctx]).start()
        elif event_type == EventType.AFTER_CHECK_CLIENT_RESOURCES:
            client_resource_result = fl_ctx.get_prop(FLContextKey.RESOURCE_CHECK_RESULT)
            if client_resource_result:
                for site_name, check_result in client_resource_result.items():
                    is_resource_enough, reason = check_result
                    if (
                        not is_resource_enough
                        and reason.startswith(CC_VERIFY_ERROR)
                        and self.critical_level == SHUTDOWN_SYSTEM
                    ):
                        threading.Thread(target=self._shutdown_system, args=[reason, fl_ctx]).start()
                        break
        elif event_type == EventType.SUBMIT_JOB:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META, {})
            byoc = job_meta.get(AppValidationKey.BYOC, False)
            if byoc:
                fl_ctx.set_prop(
                    key=FLContextKey.JOB_BLOCK_REASON, value="BYOC job not allowed for CC", sticky=False, private=True
                )

    def _setup_cc_authorizers(self, fl_ctx):
        engine = fl_ctx.get_engine()
        for conf in self.cc_issuers_conf:
            issuer_id = conf.get(CC_ISSUER_ID)
            # TODO: should we make expiration an instance variable of the issuer?
            expiration = conf.get(TOKEN_EXPIRATION)
            issuer = engine.get_component(issuer_id)
            if not isinstance(issuer, CCAuthorizer):
                raise RuntimeError(f"cc_issuer_id {issuer_id} must be a CCAuthorizer, but got {type(issuer).__name__}")
            self.cc_issuers[issuer] = expiration

        for v_id in self.cc_verifier_ids:
            verifier = engine.get_component(v_id)
            if not isinstance(verifier, CCAuthorizer):
                raise RuntimeError(f"cc_verifier_id {v_id} must be a CCAuthorizer, but got {type(verifier).__name__}")
            namespace = verifier.get_namespace()
            if namespace in self.cc_verifiers.keys():
                raise RuntimeError(f"Authorizer with namespace: {namespace} already exist.")
            self.cc_verifiers[namespace] = verifier

    def _prepare_cc_info(self, fl_ctx: FLContext):
        # client side
        self._ensure_fresh_tokens(force=True)

        if not self.token_submitted:
            site_cc_info = self.participant_cc_info[self.site_name]
            cc_info = self._get_participant_tokens(site_cc_info)
            fl_ctx.set_prop(key=CC_INFO, value=cc_info, sticky=False, private=False)
            self.logger.info("Sent the CC-tokens to server.")
            self.token_submitted = True

    def _add_client_token(self, fl_ctx: FLContext):
        # server side
        peer_ctx = fl_ctx.get_peer_context()
        token_owner = peer_ctx.get_identity_name()
        peer_cc_info = peer_ctx.get_prop(CC_INFO)

        if peer_cc_info:
            self.participant_cc_info[token_owner] = peer_cc_info
            self.logger.info(f"Added CC client: {token_owner}")

        if not self.verify_time or time.time() - self.verify_time > self.verify_frequency:
            self._verify_running_jobs(fl_ctx)

    def _verify_running_jobs(self, fl_ctx):
        engine = fl_ctx.get_engine()
        run_processes = engine.run_processes
        running_jobs = list(run_processes.keys())
        with self.lock:
            for job_id in running_jobs:
                job_participants = run_processes[job_id].get(RunProcessKey.PARTICIPANTS)
                participants = []
                for _, client in job_participants.items():
                    participants.append(client.name)

                self._ensure_fresh_tokens()
                participants_tokens = self._collect_participants_tokens(participants)
                err = self._validate_participants_tokens(participants_tokens)
                if err:
                    if self.critical_level == SHUTDOWN_JOB:
                        # maybe shutdown the whole system here. leave the user to define the action
                        engine.job_runner.stop_run(job_id, fl_ctx)
                        self.logger.info(f"Stop Job: {job_id} with CC verification error: {err} ")
                    else:
                        threading.Thread(target=self._shutdown_system, args=[err, fl_ctx]).start()

        self.verify_time = time.time()

    def _remove_client_token(self, fl_ctx: FLContext):
        # server side
        peer_ctx = fl_ctx.get_peer_context()
        token_owner = peer_ctx.get_identity_name()
        if token_owner in self.participant_cc_info.keys():
            self.participant_cc_info.pop(token_owner)
            self.logger.info(f"Removed CC client: {token_owner}")

    def _generate_tokens(self, fl_ctx: FLContext) -> str:
        # both server and client sides
        self.site_name = fl_ctx.get_identity_name()

        self.participant_cc_info[self.site_name] = []
        for issuer, expiration in self.cc_issuers.items():
            try:
                my_token = issuer.generate()
                namespace = issuer.get_namespace()

                if not isinstance(expiration, int):
                    raise ValueError(f"token_expiration value must be int, but got {type(expiration).__name__}")
                if not my_token:
                    return f"{issuer} failed to get CC token"

                self.logger.info(f"site: {self.site_name} namespace: {namespace} got the token: {my_token}")
                site_cc_info = {
                    CC_TOKEN: my_token,
                    CC_ISSUER: issuer,
                    CC_NAMESPACE: namespace,
                    TOKEN_GENERATION_TIME: time.time(),
                    TOKEN_EXPIRATION: int(expiration),
                    CC_TOKEN_VALIDATED: True,
                }
                self.participant_cc_info[self.site_name].append(site_cc_info)
                self.token_submitted = False
            except CCTokenGenerateError:
                raise RuntimeError(f"{issuer} failed to generate CC token.")

        return ""

    def _client_to_check_participant_token(self, fl_ctx: FLContext) -> str:
        # Client side
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx is None:
            return f"Empty peer context in {self.site_name=}"
        participants_tokens = peer_ctx.get_prop(PEER_CTX_CC_TOKEN, None)
        if not participants_tokens:
            return "missing PEER_CTX_CC_TOKEN prop in peer context"

        if not isinstance(participants_tokens, dict):
            return f"bad PEER_CTX_CC_TOKEN prop in peer context: must be a dict but got {type(participants_tokens)}"

        if not participants_tokens:
            return ""

        return self._validate_participants_tokens(participants_tokens)

    def _server_to_check_client_token(self, fl_ctx: FLContext) -> str:
        # Server side
        participants = fl_ctx.get_prop(FLContextKey.JOB_PARTICIPANTS)
        if not participants:
            return f"missing '{FLContextKey.JOB_PARTICIPANTS}' prop in fl_ctx"

        if not isinstance(participants, list):
            return f"bad value for {FLContextKey.JOB_PARTICIPANTS} in fl_ctx: expect list bot got {type(participants)}"

        for p in participants:
            if not isinstance(p, str):
                return f"bad value for {FLContextKey.JOB_PARTICIPANTS} in fl_ctx: expect list of str but got list of {type(p)}"

        # server side to collect tokens from all participants including itself
        # must ask each participant to generate new tokens since this method
        # is called when a job is to be scheduled
        self._ensure_fresh_tokens(force=True)
        participants_tokens = self._collect_participants_tokens(participants)
        err = self._validate_participants_tokens(participants_tokens)
        if err:
            return err

        fl_ctx.set_prop(key=PEER_CTX_CC_TOKEN, value=participants_tokens, sticky=False, private=False)
        self.logger.info(f"{self.site_name=} set PEER_CTX_CC_TOKEN with {participants_tokens=}")
        return ""

    def _collect_participants_tokens(self, participants: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Collects tokens from all participants including itself.

        Args:
            participants: list of participant names

        Returns:
            dict of participant name to list of tokens
        """
        # server side to collect tokens from all participants including itself

        participant_tokens = {}
        site_cc_info = self.participant_cc_info[self.site_name]
        participant_tokens[self.site_name] = self._get_participant_tokens(site_cc_info)

        for p in participants:
            if p == self.site_name:
                continue
            # if p not in self.participant_cc_info:
            #     return f"no token available for participant {p}"
            if self.participant_cc_info.get(p):
                participant_tokens[p] = self._get_participant_tokens(self.participant_cc_info[p])
            else:
                participant_tokens[p] = [{CC_TOKEN: "", CC_NAMESPACE: ""}]
        return participant_tokens

    def _get_participant_tokens(self, site_cc_info):
        cc_info = []
        for i in site_cc_info:
            namespace = i.get(CC_NAMESPACE)
            token = i.get(CC_TOKEN)
            cc_info.append({CC_TOKEN: token, CC_NAMESPACE: namespace, CC_TOKEN_VALIDATED: False})
        return cc_info

    def _ensure_fresh_tokens(self, force=False):
        """Refresh CC tokens for the current site by requesting each issuer to generate a new token.

        If `force` is True, generates and replaces all tokens regardless of expiration.
        Otherwise, only tokens that have expired are refreshed. The token information and
        generation time are updated in place, and all refresh events are logged for
        auditing purposes.

        Args:
            force (bool): If True, generates a new token for every issuer. If False, only
                refreshes tokens that have expired.

        Returns:
            None

        Side Effects:
            Updates self.participant_cc_info[self.site_name] in place and resets
            self.token_submitted when any token is refreshed.
        """
        site_cc_info = self.participant_cc_info[self.site_name]
        for i in site_cc_info:
            issuer = i.get(CC_ISSUER)
            token_generate_time = i.get(TOKEN_GENERATION_TIME)
            expiration = i.get(TOKEN_EXPIRATION)
            if force or time.time() - token_generate_time > expiration:
                token = issuer.generate()
                i[CC_TOKEN] = token
                i[TOKEN_GENERATION_TIME] = time.time()
                self.logger.info(
                    f"site: {self.site_name} namespace: {issuer.get_namespace()} got a new CC token: {token}"
                )

                self.token_submitted = False

    def _validate_participants_tokens(self, participants_tokens: Dict[str, List[Dict[str, str]]]) -> str:
        self.logger.debug(f"Validating participant tokens {participants_tokens=}")
        _, invalid_participant_list = self._verify_participants_tokens(participants_tokens)
        if invalid_participant_list:
            invalid_participant_string = ",".join(invalid_participant_list)
            return f"Participant {invalid_participant_string}" + CC_VERIFICATION_FAILED
        else:
            return ""

    def _verify_participants_tokens(
        self, participants_tokens: Dict[str, List[Dict[str, str]]]
    ) -> Tuple[Dict[str, bool], List[str]]:
        """Verifies tokens for all participants.

        Args:
            participants_tokens: dict of participant name to list of tokens

        Returns:
            tuple of (result, invalid_participant_list)
            result: dict of participant name to bool
            invalid_participant_list: list of invalid participants
        """
        result = {}
        invalid_participant_list = []
        if not participants_tokens:
            return result, invalid_participant_list
        for k, cc_info in participants_tokens.items():
            if k not in self.cc_enabled_sites:
                result[k] = True
                continue
            for v in cc_info:
                token = v.get(CC_TOKEN, "")
                namespace = v.get(CC_NAMESPACE, "")
                verifier = self.cc_verifiers.get(namespace, None)
                try:
                    if verifier and verifier.verify(token):
                        result[k + "." + namespace] = True
                    else:
                        invalid_participant_list.append(k + " namespace: {" + namespace + "}")
                except CCTokenVerifyError:
                    invalid_participant_list.append(k + " namespace: {" + namespace + "}")
        self.logger.info(f"CC - results from _verify_participants_tokens: {result}, {invalid_participant_list=}")
        return result, invalid_participant_list

    def _block_job(self, reason: str, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID, "")
        self.log_error(fl_ctx, f"Job {job_id} is blocked: {reason}")
        fl_ctx.set_prop(key=FLContextKey.JOB_BLOCK_REASON, value=CC_VERIFY_ERROR + reason, sticky=False)
        fl_ctx.set_prop(key=FLContextKey.AUTHORIZATION_RESULT, value=False, sticky=False)

    def _shutdown_system(self, reason: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        run_processes = engine.run_processes
        running_jobs = list(run_processes.keys())
        for job_id in running_jobs:
            engine.job_runner.stop_run(job_id, fl_ctx)

        conn = Connection(app_ctx=engine, props={ConnProps.ADMIN_SERVER: engine.server.admin_server})
        cmd = TrainingCommandModule()
        args = ["shutdown", "all"]
        cmd.validate_command_targets(conn, args[1:])
        cmd.shutdown(conn, args)

        self.logger.error(f"CC system shutdown! due to reason: {reason}")
