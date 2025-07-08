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
import socket
import time
import traceback
import uuid

from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import make_reply as make_cellnet_reply
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.f3.cellnet.defs import ReturnCode as F3ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import CellChannel, CellChannelTopic, CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, IdentityVerifier, TokenVerifier, load_crt_bytes


def _get_client_ip():
    """Return localhost IP.

    More robust than ``socket.gethostbyname(socket.gethostname())``. See
    https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib/28950776#28950776
    for more details.

    Returns:
        The host IP

    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # doesn't even have to be reachable
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class Authenticator:
    def __init__(
        self,
        cell: Cell,
        project_name: str,
        client_name: str,
        client_type: str,
        expected_sp_identity: str,
        secure_mode: bool,
        root_cert_file: str,
        private_key_file: str,
        cert_file: str,
        msg_timeout: float,
        retry_interval: float,
        timeout=None,
    ):
        """Authenticator is to be used to register a client to the Server.

        Args:
            cell: the communication cell
            project_name: name of the project
            client_name: name of the client
            client_type: type of the client: regular or relay
            expected_sp_identity: identity of the service provider (i.e. server)
            secure_mode: whether the project is in secure training mode
            root_cert_file: file path of the root cert
            private_key_file: file path of the private key
            cert_file: file path of the client's certificate
            msg_timeout: timeout for authentication messages
            retry_interval: interval between tries
            timeout: overall timeout for the authentication.
        """
        self.cell = cell
        self.project_name = project_name
        self.client_name = client_name
        self.client_type = client_type
        self.expected_sp_identity = expected_sp_identity
        self.root_cert_file = root_cert_file
        self.private_key_file = private_key_file
        self.cert_file = cert_file
        self.msg_timeout = msg_timeout
        self.retry_interval = retry_interval
        self.secure_mode = secure_mode
        self.timeout = timeout
        self.logger = get_obj_logger(self)

    def _challenge_server(self):
        # ask server for its info and make sure that it matches expected host
        my_nonce = str(uuid.uuid4())
        headers = {IdentityChallengeKey.COMMON_NAME: self.client_name, IdentityChallengeKey.NONCE: my_nonce}
        challenge = new_cell_message(headers, None)
        result = self.cell.send_request(
            target=FQCN.ROOT_SERVER,
            channel=CellChannel.SERVER_MAIN,
            topic=CellChannelTopic.Challenge,
            request=challenge,
            timeout=self.msg_timeout,
            optional=True,
        )
        return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
        error = result.get_header(MessageHeaderKey.ERROR, "")
        self.logger.debug(f"challenge result: {return_code} {error}")
        if return_code != ReturnCode.OK:
            if return_code in [ReturnCode.TARGET_UNREACHABLE, ReturnCode.COMM_ERROR]:
                # trigger retry
                return None, None
            err = result.get_header(MessageHeaderKey.ERROR, "")
            raise FLCommunicationError(f"failed to challenge server: {return_code}: {err}")

        reply = result.payload
        assert isinstance(reply, Shareable)
        server_nonce = reply.get(IdentityChallengeKey.NONCE)
        cert_bytes = reply.get(IdentityChallengeKey.CERT)
        server_cert = load_crt_bytes(cert_bytes)
        server_signature = reply.get(IdentityChallengeKey.SIGNATURE)
        server_cn = reply.get(IdentityChallengeKey.COMMON_NAME)

        if server_cn != self.expected_sp_identity:
            raise FLCommunicationError(
                f"expected server identity is '{self.expected_sp_identity}' but got '{server_cn}'"
            )

        # Use IdentityVerifier to validate:
        # - the server cert can be validated with the root cert. Note that all sites have the same root cert!
        # - the asserted CN matches the CN on the server cert
        # - signature received from the server is valid
        id_verifier = IdentityVerifier(root_cert_file=self.root_cert_file)
        id_verifier.verify_common_name(
            asserter_cert=server_cert, asserted_cn=server_cn, nonce=my_nonce, signature=server_signature
        )

        self.logger.info(f"verified server identity '{self.expected_sp_identity}'")
        return server_nonce, TokenVerifier(server_cert)

    def authenticate(self, shared_fl_ctx: FLContext, abort_signal: Signal):
        """Register the client with the FLARE Server.

        Note that the client no longer needs to be directly connected with the Server!

        Since the client may be connected with the Server indirectly (e.g. via bridge nodes or proxy), in the secure
        mode, the client authentication cannot be based on the connection's TLS cert. Instead, the server and the
        client will explicitly authenticate each other using their provisioned PKI credentials, as follows:

        1. Make sure that the Server is authentic. The client sends a Challenge request with a random nonce.
        The server is expected to return the following in its reply:
            - its cert and common name (Server_CN)
            - signature on the received client nonce + Server_CN
            - a random Server Nonce. This will be used for the server to validate the client's identity in the
            Registration request.

        The client then validates to make sure:
            - the Server_CN is the same as presented in the server cert
            - the Server_CN is the same as configured in the client's config (fed_client.json)
            - the signature is valid

        2. Client sends Registration request that contains:
            - client cert and common name (Client_CN)
            - signature on the received Server Nonce + Client_CN

        The Server then validates to make sure:
            - the Client_CN is the same as presented in the client cert
            - the signature is valid

        NOTE: we do not explicitly validate certs' expiration time. This is because currently the same certs are
        also used for SSL connections, which already validate expiration.

        Args:
            shared_fl_ctx: the FLContext content to be shared with peer
            abort_signal: signal to notify abort

        Returns: A tuple of (token, token_signature, ssid, token_verifier)

        """
        local_ip = _get_client_ip()
        shareable = Shareable()
        shareable.set_peer_context(shared_fl_ctx)

        token_verifier = None
        if self.secure_mode:
            # explicitly authenticate with the Server
            start_time = time.time()
            while True:
                server_nonce, token_verifier = self._challenge_server()

                if abort_signal.triggered:
                    return None, None, None, None

                if server_nonce is None:
                    # retry
                    self.logger.info(f"re-challenge after {self.retry_interval} seconds")

                    if self.timeout and time.time() - start_time > self.timeout:
                        raise FLCommunicationError(f"cannot connect to server for {self.timeout} seconds")

                    time.sleep(self.retry_interval)
                else:
                    break

            id_asserter = IdentityAsserter(private_key_file=self.private_key_file, cert_file=self.cert_file)
            cn_signature = id_asserter.sign_common_name(nonce=server_nonce)
            shareable[IdentityChallengeKey.CERT] = id_asserter.cert_data
            shareable[IdentityChallengeKey.SIGNATURE] = cn_signature
            shareable[IdentityChallengeKey.COMMON_NAME] = id_asserter.cn
            self.logger.debug(f"sent identity info for client {self.client_name}")

        headers = {
            CellMessageHeaderKeys.CLIENT_NAME: self.client_name,
            CellMessageHeaderKeys.CLIENT_TYPE: self.client_type,
            CellMessageHeaderKeys.CLIENT_IP: local_ip,
            CellMessageHeaderKeys.PROJECT_NAME: self.project_name,
        }
        login_message = new_cell_message(headers, shareable)

        self.logger.debug("Trying to register with server ...")
        start_time = time.time()
        while True:
            try:
                result = self.cell.send_request(
                    target=FQCN.ROOT_SERVER,
                    channel=CellChannel.SERVER_MAIN,
                    topic=CellChannelTopic.Register,
                    request=login_message,
                    timeout=self.msg_timeout,
                    optional=True,
                )

                if not isinstance(result, Message):
                    raise FLCommunicationError(f"expect result to be Message but got {type(result)}")

                return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
                self.logger.debug(f"register RC: {return_code}")
                if return_code == ReturnCode.UNAUTHENTICATED:
                    reason = result.get_header(MessageHeaderKey.ERROR)
                    self.logger.error(f"registration rejected: {reason}")
                    raise FLCommunicationError("error:client_registration " + reason)

                payload = result.payload
                if not isinstance(payload, dict):
                    raise FLCommunicationError(f"expect payload to be dict but got {type(payload)}")

                token = payload.get(CellMessageHeaderKeys.TOKEN)
                token_signature = payload.get(CellMessageHeaderKeys.TOKEN_SIGNATURE, "NA")
                ssid = payload.get(CellMessageHeaderKeys.SSID)
                if not token and not abort_signal.triggered:
                    if self.timeout and time.time() - start_time > self.timeout:
                        # timed out
                        raise FLCommunicationError(f"cannot authenticate to server for {self.timeout} seconds")

                    time.sleep(self.retry_interval)
                else:
                    break

            except Exception as ex:
                traceback.print_exc()
                raise FLCommunicationError("error:client_registration", ex)

        # make sure token_verifier works
        if token_verifier:
            if not isinstance(token_verifier, TokenVerifier):
                raise RuntimeError(f"expect token_verifier to be TokenVerifier but got {type(token_verifier)}")

        if token_verifier and token_signature:
            valid = token_verifier.verify(client_name=self.client_name, token=token, signature=token_signature)
            if valid:
                self.logger.info("Verified received token and signature successfully")
            else:
                raise RuntimeError("invalid token or verifier!")

        return token, token_signature, ssid, token_verifier


def validate_auth_headers(message: CellMessage, token_verifier: TokenVerifier, logger):
    """Validate auth headers from messages that go through the server.

    Args:
        message: the message to validate
        token_verifier: the TokenVerifier to be used to verify the token and signature

    Returns:
    """
    headers = message.headers
    logger.debug(f"**** _validate_auth_headers: {headers=}")
    topic = message.get_header(MessageHeaderKey.TOPIC)
    channel = message.get_header(MessageHeaderKey.CHANNEL)

    origin = message.get_header(MessageHeaderKey.ORIGIN)

    if topic in [CellChannelTopic.Register, CellChannelTopic.Challenge] and channel == CellChannel.SERVER_MAIN:
        # skip: client not registered yet
        logger.debug(f"skip special message {topic=} {channel=}")
        return None

    client_name = message.get_header(CellMessageHeaderKeys.CLIENT_NAME)
    err_text = f"unauthenticated msg ({channel=} {topic=}) received from {origin}"
    if not client_name:
        err = "missing client name"
        logger.error(f"{err_text}: {err}")
        return make_cellnet_reply(rc=F3ReturnCode.UNAUTHENTICATED, error=err)

    token = message.get_header(CellMessageHeaderKeys.TOKEN)
    if not token:
        err = "missing auth token"
        logger.error(f"{err_text}: {err}")
        return make_cellnet_reply(rc=F3ReturnCode.UNAUTHENTICATED, error=err)

    signature = message.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE)
    if not signature:
        err = "missing auth token signature"
        logger.error(f"{err_text}: {err}")
        return make_cellnet_reply(rc=F3ReturnCode.UNAUTHENTICATED, error=err)

    if not token_verifier.verify(client_name, token, signature):
        err = "invalid auth token signature"
        logger.error(f"{err_text}: {err}")
        return make_cellnet_reply(rc=F3ReturnCode.UNAUTHENTICATED, error=err)

    # all good
    logger.debug(f"auth headers valid from {origin}: {topic=} {channel=}")
    return None
