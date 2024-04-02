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
from unittest.mock import Mock

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_opt.confidential_computing.cc_manager import (
    CC_INFO,
    CC_NAMESPACE,
    CC_TOKEN,
    CC_TOKEN_VALIDATED,
    CC_VERIFICATION_FAILED,
    CCManager,
)
from nvflare.app_opt.confidential_computing.tdx_authorizer import TDX_NAMESPACE, TDXAuthorizer

VALID_TOKEN = "valid_token"
INVALID_TOKEN = "invalid_token"


class TestCCManager:
    def setup_method(self, method):
        issues_conf = [{"issuer_id": "tdx_authorizer", "token_expiration": 250}]

        verify_ids = (["tdx_authorizer"],)
        self.cc_manager = CCManager(issues_conf, verify_ids)

    def test_authorizer_setup(self):

        fl_ctx, tdx_authorizer = self._setup_authorizers()

        assert self.cc_manager.cc_issuers == {tdx_authorizer: 250}
        assert self.cc_manager.cc_verifiers == {TDX_NAMESPACE: tdx_authorizer}

    def _setup_authorizers(self):
        fl_ctx = Mock(spec=FLContext)
        fl_ctx.get_identity_name.return_value = "server"
        engine = Mock(spec=ServerEngineSpec)
        fl_ctx.get_engine.return_value = engine

        tdx_authorizer = Mock(spec=TDXAuthorizer)
        tdx_authorizer.get_namespace.return_value = TDX_NAMESPACE
        tdx_authorizer.verify = self._verify_token
        engine.get_component.return_value = tdx_authorizer
        self.cc_manager._setup_cc_authorizers(fl_ctx)

        tdx_authorizer.generate.return_value = VALID_TOKEN
        self.cc_manager._generate_tokens(fl_ctx)

        return fl_ctx, tdx_authorizer

    def _verify_token(self, token):
        if token == VALID_TOKEN:
            return True
        else:
            return False

    def test_add_client_token(self):

        cc_info1, cc_info2 = self._add_failed_tokens()

        assert self.cc_manager.participant_cc_info["client1"] == cc_info1
        assert self.cc_manager.participant_cc_info["client2"] == cc_info2

    def _add_failed_tokens(self):
        self.cc_manager._verify_running_jobs = Mock()
        client_name = "client1"
        valid_token = VALID_TOKEN
        cc_info1, fl_ctx = self._add_client_token(client_name, valid_token)
        self.cc_manager._add_client_token(fl_ctx)

        client_name = "client2"
        valid_token = INVALID_TOKEN
        cc_info2, fl_ctx = self._add_client_token(client_name, valid_token)
        self.cc_manager._add_client_token(fl_ctx)
        return cc_info1, cc_info2

    def test_verification_success(self):

        self._setup_authorizers()

        self.cc_manager._verify_running_jobs = Mock()

        self.cc_manager._verify_running_jobs = Mock()
        client_name = "client1"
        valid_token = VALID_TOKEN
        cc_info1, fl_ctx = self._add_client_token(client_name, valid_token)
        self.cc_manager._add_client_token(fl_ctx)

        client_name = "client2"
        valid_token = VALID_TOKEN
        cc_info2, fl_ctx = self._add_client_token(client_name, valid_token)
        self.cc_manager._add_client_token(fl_ctx)

        self.cc_manager._handle_expired_tokens = Mock()

        err, participant_tokens = self.cc_manager._verify_participants(["client1", "client2"])

        assert not err

    def test_verification_failed(self):

        self._setup_authorizers()

        self.cc_manager._verify_running_jobs = Mock()
        self._add_failed_tokens()
        self.cc_manager._handle_expired_tokens = Mock()

        err, participant_tokens = self.cc_manager._verify_participants(["client1", "client2"])

        assert "client2" in err
        assert CC_VERIFICATION_FAILED in err

    def _add_client_token(self, client_name, valid_token):
        peer_ctx = FLContext()
        cc_info = [{CC_TOKEN: valid_token, CC_NAMESPACE: TDX_NAMESPACE, CC_TOKEN_VALIDATED: False}]
        peer_ctx.set_prop(CC_INFO, cc_info)
        peer_ctx.set_prop(ReservedKey.IDENTITY_NAME, client_name)
        fl_ctx = Mock(spec=FLContext)
        fl_ctx.get_peer_context.return_value = peer_ctx
        return cc_info, fl_ctx
