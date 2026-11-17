# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Exercise site ownership after transport authentication, against a real SuperLink.

This probe supplies the private authenticated-client property at the AuxRunner
boundary; transport authentication itself is outside this probe.
"""

import os
import subprocess
import sys
from base64 import b64encode
from pathlib import Path
from unittest.mock import Mock

from flwr.proto.fleet_pb2 import (
    ActivateNodeRequest,
    ActivateNodeResponse,
    DeactivateNodeRequest,
    RegisterNodeFleetRequest,
)
from flwr.supercore.date import now
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes, sign_message
from flwr.supercore.runtime_version_compatibility import RuntimeVersionMetadata

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.tie.controller import _ClientStatus
from nvflare.app_common.tie.defs import Constant
from nvflare.app_opt.flower.applet import _wait_for_superlink
from nvflare.app_opt.flower.connectors.http_server_connector import HttpServerConnector
from nvflare.app_opt.flower.controller import FlowerController
from nvflare.app_opt.flower.http_transport import FleetHttpClient
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.private.aux_runner import AuxRunner
from tests.process_test_utils import OwnedProcesses


def run(workspace):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    private_key, public_key = generate_key_pairs()
    key_bytes = public_key_to_bytes(public_key)

    def signed_request(path, message):
        timestamp = now().isoformat()
        headers = dict(RuntimeVersionMetadata.from_local_component("SuperNode").as_metadata())
        headers.update(
            {
                "content-type": "application/protobuf",
                "flwr-public-key": b64encode(key_bytes).decode("ascii"),
                "flwr-timestamp": timestamp,
                "flwr-signature": b64encode(sign_message(private_key, timestamp.encode("ascii"))).decode("ascii"),
            }
        )
        return Shareable({"method": "POST", "path": path, "headers": headers, "body": message.SerializeToString()})

    controller = FlowerController()
    controller.connector = HttpServerConnector()
    controller.abort_signal = Signal()
    controller.client_statuses = {site: _ClientStatus() for site in ("site-1", "site-2")}
    runner = AuxRunner(Mock())
    runner.register_aux_message_handler("flower.security", controller._handle_app_request)

    def dispatch(request, authenticated_site, claimed_site):
        fl_ctx = FLContext()
        if authenticated_site is not None:
            fl_ctx.set_prop(FLContextKey.CLIENT_NAME, authenticated_site, private=True, sticky=False)
        request.set_header(Constant.MSG_KEY_OP, "request")
        # Also try injecting the private property's name into sender-controlled
        # public peer properties. AuxRunner must keep the two contexts separate.
        request.set_peer_props({ReservedKey.IDENTITY_NAME: claimed_site, FLContextKey.CLIENT_NAME: claimed_site})
        return runner.dispatch("flower.security", request, fl_ctx)

    port = get_open_ports(1)[0]
    address = f"127.0.0.1:{port}"
    environment = dict(os.environ, FLWR_HOME=str(workspace / "flower-home"))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    with (workspace / "superlink.log").open("w") as log, OwnedProcesses() as owner:
        client = FleetHttpClient(address)
        owner.callbacks.append(client.close)
        controller.connector.http_client = client
        process = owner.start(
            [
                str(Path(sys.executable).with_name("flower-superlink")),
                "--insecure",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--disable-runtime-dependency-installation",
            ],
            cwd=workspace,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        _wait_for_superlink(address, process, timeout=45)
        registration = signed_request("/v1/fleet/register-node", RegisterNodeFleetRequest(public_key=key_bytes))
        assert dispatch(registration, "site-1", "site-1")["status"] == 200
        activation = signed_request(
            "/v1/fleet/activate-node", ActivateNodeRequest(public_key=key_bytes, heartbeat_interval=30)
        )
        reply = dispatch(activation, "site-1", "site-1")
        assert reply["status"] == 200, reply
        node_id = ActivateNodeResponse.FromString(reply["body"]).node_id
        assert node_id

        # Keep exactly the same signed Flower request while varying only the
        # NVFlare identities. The owner can still deactivate after the denials.
        deactivation = signed_request("/v1/fleet/deactivate-node", DeactivateNodeRequest(node_id=node_id))
        for authenticated_site, claimed_site in (("site-2", "site-2"), ("site-2", "site-1"), (None, "site-1")):
            reply = dispatch(deactivation, authenticated_site, claimed_site)
            assert reply["status"] == 403, (authenticated_site, claimed_site, reply)
        reply = dispatch(deactivation, "site-1", "site-1")
        assert reply["status"] == 200, reply
    print("FLOWER_SITE_OWNERSHIP_OK")


if __name__ == "__main__":
    run(sys.argv[1])
