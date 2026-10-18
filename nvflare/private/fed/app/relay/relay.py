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

import argparse
import json
import logging
import os
import sys
import threading

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey, ReservedKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import SSL_ROOT_CERT, enhance_credential_info
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.sec.authn import add_authentication_headers, add_server_path_reply_authentication_headers
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.config_service import ConfigService, search_file
from nvflare.fuel.utils.log_utils import configure_logging
from nvflare.fuel.utils.url_utils import make_url
from nvflare.private.defs import ClientType
from nvflare.private.fed.authenticator import Authenticator, validate_auth_headers
from nvflare.private.fed.utils.identity_utils import TokenVerifier


class CellnetMonitor:
    def __init__(self, stop_event: threading.Event, workspace: str):
        self.stop_event = stop_event
        self.workspace = workspace

    def cellnet_stopped(self):
        touch_file = os.path.join(self.workspace, WorkspaceConstants.SHUTDOWN_FILE)
        with open(touch_file, "a"):
            os.utime(touch_file, None)
        self.stop_event.set()


class _RelayAuth:
    def __init__(self, authenticator, shared_fl_ctx, auth_result, stop_event, logger):
        self.authenticator = authenticator
        # Recovery is bounded independently of startup authentication retries.
        self.authenticator.timeout = 5.0
        self.shared_fl_ctx = shared_fl_ctx
        self.auth_result = auth_result
        self.stop_event = stop_event
        self.logger = logger
        self.refresh_needed = threading.Event()

    def add_request_headers(self, message):
        add_authentication_headers(message, self.authenticator.client_name, *self.auth_result[:3])

    def add_reply_headers(self, message):
        add_server_path_reply_authentication_headers(message, self.authenticator.client_name, *self.auth_result[:3])

    def validate(self, message, **kwargs):
        result = _validate_auth_headers(message, self.auth_result[3], self.logger, **kwargs)
        if result is not None:
            # The receive path must remain fail-closed and cannot wait for a
            # challenge reply on the same connection. Recover in the main loop.
            self.refresh_needed.set()
        return result

    def refresh(self):
        if not self.refresh_needed.is_set() or self.stop_event.is_set():
            return
        self.refresh_needed.clear()
        try:
            _, verifier = self.authenticator.challenge_server()
            if verifier is None or self.stop_event.is_set():
                return
            if verifier.verify(self.authenticator.client_name, *self.auth_result[:2], log_error=False):
                return  # Same-key outages do not require a replacement token.
            auth_result = self.authenticator.authenticate(self.shared_fl_ctx, Signal())
            if self.stop_event.is_set():
                return
            self.auth_result = auth_result
            self.logger.info("Refreshed relay authentication after server key replacement")
        except Exception:
            # Keep the last authenticated state; a later rejected message can
            # request another bounded attempt. Never log credential material.
            self.logger.warning("Could not refresh relay server authentication")


class _ConfigKey:
    PROJECT_NAME = "project_name"
    SERVER_IDENTITY = "server_identity"
    IDENTITY = "identity"
    AUTH_IDENTITY = "auth_identity"
    AUTH_IDENTITY_MAP = "auth_identity_map"
    CONNECT_TO = "connect_to"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--relay_config", "-s", type=str, help="relay config json file", required=True)
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    args = parser.parse_args()
    return args


def main(args):
    workspace = Workspace(root_dir=args.workspace)
    for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
        try:
            f = workspace.get_file_path_in_root(name)
            if os.path.exists(f):
                os.remove(f)
        except Exception as ex:
            print(f"Could not remove file '{name}': {ex}.  Please check your system before starting FL.")
            sys.exit(-1)

    configure_logging(workspace)
    logger = logging.getLogger()

    relay_config_file = workspace.get_file_path_in_startup(args.relay_config)
    with open(relay_config_file, "rt") as f:
        relay_config = json.load(f)

    if not isinstance(relay_config, dict):
        raise RuntimeError(f"invalid relay config file {args.relay_config}")

    project_name = relay_config.get(_ConfigKey.PROJECT_NAME)
    if not project_name:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing {_ConfigKey.PROJECT_NAME}")

    server_identity = relay_config.get(_ConfigKey.SERVER_IDENTITY)
    if not server_identity:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing {_ConfigKey.SERVER_IDENTITY}")

    my_identity = relay_config.get(_ConfigKey.IDENTITY)
    if not my_identity:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing {_ConfigKey.IDENTITY}")

    my_auth_identity = relay_config.get(_ConfigKey.AUTH_IDENTITY, my_identity)

    parent = relay_config.get(_ConfigKey.CONNECT_TO)
    if not parent:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing {_ConfigKey.CONNECT_TO}")

    parent_address = parent.get(ConnPropKey.ADDRESS)
    if not parent_address:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.address")

    parent_scheme = parent.get(ConnPropKey.SCHEME)
    if not parent_scheme:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.scheme")

    parent_fqcn = parent.get(ConnPropKey.FQCN)
    if not parent_fqcn:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.fqcn")

    parent_identity = parent.get(_ConfigKey.AUTH_IDENTITY, parent.get(_ConfigKey.IDENTITY))
    if not parent_identity:
        if parent_fqcn == FQCN.ROOT_SERVER:
            parent_identity = server_identity
        else:
            parent_identity = FQCN.split(parent_fqcn)[-1]

    auth_identity_map = {parent_fqcn: parent_identity}
    configured_identity_map = relay_config.get(_ConfigKey.AUTH_IDENTITY_MAP)
    if configured_identity_map:
        auth_identity_map.update(configured_identity_map)

    cmd_vars = parse_vars(args.set)
    secure_train = cmd_vars.get("secure_train", False)
    logger.debug(f"{cmd_vars=} {secure_train=}")

    stop_event = threading.Event()
    monitor = CellnetMonitor(stop_event, args.workspace)

    ConfigService.initialize(
        section_files={},
        config_path=[args.workspace],
    )

    root_cert_path = search_file(SSL_ROOT_CERT, args.workspace)
    if not root_cert_path:
        raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {args.workspace}")

    credentials = {
        DriverParams.CA_CERT.value: root_cert_path,
    }
    enhance_credential_info(credentials)

    conn_security = parent.get(ConnPropKey.CONNECTION_SECURITY)
    secure_conn = True
    if conn_security:
        credentials[DriverParams.CONNECTION_SECURITY.value] = conn_security
        if conn_security == ConnectionSecurity.CLEAR:
            secure_conn = False
    parent_url = make_url(parent_scheme, parent_address, secure_conn)

    if parent_fqcn == FQCN.ROOT_SERVER:
        my_fqcn = my_identity
        root_url = parent_url
        parent_url = None
    else:
        my_fqcn = FQCN.join([parent_fqcn, my_identity])
        root_url = None

    flare_decomposers.register()

    cell = Cell(
        fqcn=my_fqcn,
        root_url=root_url,
        secure=secure_conn,
        credentials=credentials,
        create_internal_listener=True,
        parent_url=parent_url,
        auth_identity=my_auth_identity,
        auth_identity_map=auth_identity_map,
    )
    NetAgent(cell, agent_closed_cb=monitor.cellnet_stopped)
    cell.start()

    # authenticate
    authenticator = Authenticator(
        cell=cell,
        project_name=project_name,
        client_name=my_identity,
        client_type=ClientType.RELAY,
        expected_sp_identity=server_identity,
        secure_mode=secure_train,
        root_cert_file=credentials.get(DriverParams.CA_CERT.value),
        private_key_file=credentials.get(DriverParams.CLIENT_KEY.value),
        cert_file=credentials.get(DriverParams.CLIENT_CERT.value),
        msg_timeout=5.0,
        retry_interval=2.0,
    )

    abort_signal = Signal()
    shared_fl_ctx = FLContext()
    shared_fl_ctx.set_public_props({ReservedKey.IDENTITY_NAME: my_identity})
    auth_result = authenticator.authenticate(
        shared_fl_ctx=shared_fl_ctx,
        abort_signal=abort_signal,
    )

    _, _, ssid, token_verifier = auth_result
    relay_auth = None
    if secure_train:
        if not isinstance(token_verifier, TokenVerifier):
            raise RuntimeError(f"expect token_verifier to be TokenVerifier but got {type(token_verifier)}")

        relay_auth = _RelayAuth(authenticator, shared_fl_ctx, auth_result, stop_event, logger)
        cell.core_cell.add_outgoing_request_filter(channel="*", topic="*", cb=relay_auth.add_request_headers)
        cell.core_cell.add_outgoing_reply_filter(channel="*", topic="*", cb=relay_auth.add_reply_headers)
        cell.core_cell.add_incoming_filter(
            channel="*",
            topic="*",
            cb=relay_auth.validate,
            local_cell_fqcn=my_fqcn,
            core_cell=cell.core_cell,
        )

    logger.info(f"Successfully authenticated to {server_identity}: {ssid=}")

    # wait until stopped
    logger.info(f"Started relay {my_identity=} {my_fqcn=} {root_url=} {parent_url=} {parent_fqcn=}")
    while not stop_event.wait(2.0):
        if relay_auth:
            relay_auth.refresh()
    cell.stop()
    logger.info(f"Relay {my_fqcn} stopped.")


def _validate_auth_headers(
    message: CellMessage, token_verifier: TokenVerifier, logger, local_cell_fqcn=None, core_cell=None
):
    """Validate auth headers from messages that go through the relay.
    Args:
        message: the message to validate
        token_verifier: verifier for the token and signature
        logger: logger
        local_cell_fqcn: FQCN of this relay's cell, used to scope the cellnet bye auth bypass
        core_cell: relay CoreCell used to verify a post-server message's actual upstream endpoint
    Returns:
    """
    if core_cell and core_cell.is_server_transit_return(message):
        logger.debug("accepting sanitized server-transit message from configured upstream parent")
        return None
    return validate_auth_headers(message, token_verifier, logger, local_cell_fqcn=local_cell_fqcn)


if __name__ == "__main__":
    args = parse_arguments()
    rc = mpm.run(main_func=main, run_dir=args.workspace, args=args)
    sys.exit(rc)
