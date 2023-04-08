# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import shlex
import shutil
import socket
import ssl
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import grpc
from requests import Request, RequestException, Response, Session, codes
from requests.adapters import HTTPAdapter

from nvflare.fuel.hci.conn import ALL_END


class NVFlareConfig:
    OVERSEER = "gunicorn.conf.py"
    SERVER = "fed_server.json"
    CLIENT = "fed_client.json"
    ADMIN = "fed_admin.json"


class NVFlareRole:
    SERVER = "server"
    CLIENT = "client"
    ADMIN = "admin"


def try_write_dir(path: str):
    try:
        created = False
        if not os.path.exists(path):
            created = True
            os.makedirs(path, exist_ok=False)
        fd, name = tempfile.mkstemp(dir=path)
        with os.fdopen(fd, "w") as fp:
            fp.write("dummy")
        os.remove(name)
        if created:
            shutil.rmtree(path)
    except OSError as e:
        return e


def try_bind_address(host: str, port: int):
    """Tries to bind to address."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError as e:
        return e
    finally:
        sock.close()
    return None


def _create_http_session(ca_path=None, cert_path=None, prv_key_path=None):
    session = Session()
    adapter = HTTPAdapter(max_retries=1)
    session.mount("https://", adapter)
    if ca_path:
        session.verify = ca_path
        session.cert = (cert_path, prv_key_path)
    return session


def _send_request(
    session, api_point, headers: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None
) -> Response:
    req = Request("POST", api_point, json=payload, headers=headers)
    prepared = session.prepare_request(req)
    resp = session.send(prepared)
    return resp


def parse_overseer_agent_args(overseer_agent_conf: dict, required_args: list) -> dict:
    result = {}
    for k in required_args:
        value = overseer_agent_conf.get("args", {}).get(k)
        if value is None:
            raise Exception(f"overseer agent missing arg '{k}'.")
        result[k] = value
    return result


def construct_dummy_response(overseer_agent_args: dict) -> Response:
    psp = {"sp_end_point": overseer_agent_args["sp_end_point"], "primary": True}
    response_content = {"primary_sp": psp, "sp_list": [psp]}
    resp = Response()
    resp.status_code = 200
    resp._content = str.encode(json.dumps(response_content))
    return resp


def is_dummy_overseer_agent(overseer_agent_class: str) -> bool:
    if overseer_agent_class == "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent":
        return True
    return False


def get_required_args_for_overseer_agent(overseer_agent_class: str, role: str) -> list:
    """Gets required argument list for a specific overseer agent class."""
    if overseer_agent_class == "nvflare.ha.overseer_agent.HttpOverseerAgent":
        required_args = ["overseer_end_point", "role", "project", "name"]
        if role == NVFlareRole.SERVER:
            required_args.extend(["fl_port", "admin_port"])
        return required_args
    elif overseer_agent_class == "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent":
        required_args = ["sp_end_point"]
        return required_args
    else:
        raise Exception(f"overseer agent {overseer_agent_class} is not supported.")


def _prepare_data(args: dict):
    data = dict(role=args["role"], project=args["project"])
    if args["role"] == NVFlareRole.SERVER:
        data["sp_end_point"] = ":".join([args["name"], args["fl_port"], args["admin_port"]])
    return data


def _get_ca_cert_file_name():
    return "rootCA.pem"


def _get_cert_file_name(role: str):
    if role == NVFlareRole.SERVER:
        return "server.crt"
    return "client.crt"


def _get_prv_key_file_name(role: str):
    if role == NVFlareRole.SERVER:
        return "server.key"
    return "client.key"


def split_by_len(item, max_len):
    return [item[ind : ind + max_len] for ind in range(0, len(item), max_len)]


def check_overseer_running(
    startup: str, overseer_agent_args: dict, role: str, retry: int = 3
) -> Tuple[Optional[Response], Optional[str]]:
    """Checks if overseer is running."""
    session = _create_http_session(
        ca_path=os.path.join(startup, _get_ca_cert_file_name()),
        cert_path=os.path.join(startup, _get_cert_file_name(role)),
        prv_key_path=os.path.join(startup, _get_prv_key_file_name(role)),
    )
    data = _prepare_data(overseer_agent_args)
    try_count = 0
    retry_delay = 1
    resp = None
    err = None
    while try_count < retry:
        try:
            resp = _send_request(
                session,
                api_point=overseer_agent_args["overseer_end_point"] + "/heartbeat",
                payload=data,
            )
            if resp:
                break
        except RequestException as e:
            try_count += 1
            time.sleep(retry_delay)
            err = str(e)
    return resp, err


def check_response(resp: Optional[Response]) -> bool:
    if not resp:
        return False
    if resp.status_code != codes.ok:
        return False
    return True


def check_socket_server_running(startup: str, host: str, port: int) -> bool:
    try:
        # SSL communication
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = False

        ctx.load_verify_locations(os.path.join(startup, _get_ca_cert_file_name()))
        ctx.load_cert_chain(
            certfile=os.path.join(startup, _get_cert_file_name(NVFlareRole.CLIENT)),
            keyfile=os.path.join(startup, _get_prv_key_file_name(NVFlareRole.CLIENT)),
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with ctx.wrap_socket(sock) as secure_sock:
                secure_sock.connect((host, port))
                secure_sock.sendall(bytes(f"hello{ALL_END}", "utf-8"))
                secure_sock.recv()
    except Exception as e:
        print(e)
        return False
    return True


def check_grpc_server_running(startup: str, host: str, port: int, token=None) -> bool:
    with open(os.path.join(startup, _get_ca_cert_file_name()), "rb") as f:
        trusted_certs = f.read()
    with open(os.path.join(startup, _get_prv_key_file_name(NVFlareRole.CLIENT)), "rb") as f:
        private_key = f.read()
    with open(os.path.join(startup, _get_cert_file_name(NVFlareRole.CLIENT)), "rb") as f:
        certificate_chain = f.read()

    call_credentials = grpc.metadata_call_credentials(
        lambda context, callback: callback((("x-custom-token", token),), None)
    )
    credentials = grpc.ssl_channel_credentials(
        certificate_chain=certificate_chain, private_key=private_key, root_certificates=trusted_certs
    )

    composite_credentials = grpc.composite_channel_credentials(credentials, call_credentials)
    channel = grpc.secure_channel(target=f"{host}:{port}", credentials=composite_credentials)
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        return False
    return True


def run_command_in_subprocess(command):
    new_env = os.environ.copy()
    process = subprocess.Popen(
        shlex.split(command),
        preexec_fn=os.setsid,
        env=new_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return process
