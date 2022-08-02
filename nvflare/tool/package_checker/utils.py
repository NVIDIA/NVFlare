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

import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

from requests import Request, RequestException, Response, Session, codes
from requests.adapters import HTTPAdapter


def try_write(path: str):
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
    except socket.error as e:
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


def _parse_overseer_agent(d: dict, required_args: list):
    result = {}
    overseer_agent_path = d.get("path")
    if overseer_agent_path != "nvflare.ha.overseer_agent.HttpOverseerAgent":
        raise Exception(f"overseer agent {overseer_agent_path} is not supported.")
    for k in required_args:
        value = d.get("args", {}).get(k)
        if value is None:
            raise Exception(f"overseer agent missing arg '{k}'.")
        result[k] = value
    return result


def _prepare_data(args: dict):
    data = dict(role=args["role"], project=args["project"])
    if args["role"] == "server":
        data["sp_end_point"] = ":".join([args["name"], args["fl_port"], args["admin_port"]])
    return data


def check_overseer_running(startup: str, overseer_agent_conf: dict, role: str, retry: int = 3) -> Optional[Response]:
    """Checks if overseer is running."""
    cert_file = "server.crt" if role == "server" else "client.crt"
    prv_key_file = "server.key" if role == "server" else "client.key"
    required_args = ["overseer_end_point", "role", "project", "name"]
    if role == "server":
        required_args.extend(["fl_port", "admin_port"])

    overseer_agent_args = _parse_overseer_agent(overseer_agent_conf, required_args)
    session = _create_http_session(
        ca_path=os.path.join(startup, "rootCA.pem"),
        cert_path=os.path.join(startup, cert_file),
        prv_key_path=os.path.join(startup, prv_key_file),
    )
    data = _prepare_data(overseer_agent_args)
    try_count = 0
    retry_delay = 1
    resp = None
    while try_count < retry:
        try:
            resp = _send_request(
                session,
                api_point=overseer_agent_args["overseer_end_point"] + "/heartbeat",
                payload=data,
            )
            if resp:
                break
        except RequestException:
            try_count += 1
            time.sleep(retry_delay)
    return resp


def check_response(resp: Response) -> bool:
    if not resp:
        return False
    if resp.status_code != codes.ok:
        return False
    return True


def run_command_in_subprocess(command):
    new_env = os.environ.copy()
    process = subprocess.Popen(
        shlex.split(command),
        preexec_fn=os.setsid,
        env=new_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return process
