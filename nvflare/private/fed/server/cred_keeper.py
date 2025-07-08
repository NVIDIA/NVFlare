# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_constant import FLContextKey, SecureTrainConst
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, IdentityVerifier


class CredKeeper:

    def __init__(self):
        self.id_verifier = None
        self.id_asserter = None
        self.logger = get_obj_logger(self)
        self._lock = threading.Lock()

    def _get_server_config(self, fl_ctx: FLContext):
        server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
        if not server_config:
            self.logger.error(f"missing {FLContextKey.SERVER_CONFIG} in FL context")
            return {}

        if not isinstance(server_config, list):
            self.logger.error(f"expect server_config to be list but got {type(server_config)}")
            return {}

        server1 = server_config[0]
        if not isinstance(server1, dict):
            self.logger.error(f"expect server config data to be dict but got {type(server1)}")
            return {}
        return server1

    def get_id_verifier(self, fl_ctx: FLContext):
        with self._lock:
            if not self.id_verifier:
                config = self._get_server_config(fl_ctx)
                root_cert_file = config.get(SecureTrainConst.SSL_ROOT_CERT)
                if not root_cert_file:
                    self.logger.error(f"missing {SecureTrainConst.SSL_ROOT_CERT} in server config")
                    return None
                self.id_verifier = IdentityVerifier(root_cert_file=root_cert_file)
            return self.id_verifier

    def get_id_asserter(self, fl_ctx: FLContext):
        with self._lock:
            if not self.id_asserter:
                config = self._get_server_config(fl_ctx)
                cert_file = config.get(SecureTrainConst.SSL_CERT)

                if not cert_file:
                    self.logger.error(f"missing {SecureTrainConst.SSL_CERT} in server config")
                    return None

                private_key_file = config.get(SecureTrainConst.PRIVATE_KEY)
                self.id_asserter = IdentityAsserter(private_key_file=private_key_file, cert_file=cert_file)
            return self.id_asserter
