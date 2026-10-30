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

"""Shared PKI and loopback-port helpers for live job-certificate tests."""

import os
import socket

from nvflare.fuel.sec.cert_uri import job_ca_marker_uri
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.private.fed.utils.job_cert_utils import JobCertIssuer

# fixed ids: pytest-xdist requires identical test ids on every worker
_JOB_A = "aaaaaaaa-0000-4000-8000-00000000000a"
_JOB_B = "bbbbbbbb-0000-4000-8000-00000000000b"


def _write_pki(out_dir: str) -> dict:
    root_key, root_pub = generate_keys()
    root_cert = generate_cert(Identity("rootCA"), Identity("rootCA"), root_key, root_pub, ca=True)
    server_key, server_pub = generate_keys()
    server_cert = generate_cert(
        Identity("server"), Identity("rootCA"), root_key, server_pub, server_default_host="localhost"
    )
    site_key, site_pub = generate_keys()
    site_cert = generate_cert(
        Identity("site-1"), Identity("rootCA"), root_key, site_pub, server_default_host="localhost"
    )
    jca_key, jca_pub = generate_keys()
    jca_cert = generate_cert(
        Identity("job_ca"),
        Identity("rootCA"),
        root_key,
        jca_pub,
        ca=True,
        ca_path_length=0,
        uri_names=[job_ca_marker_uri()],
    )
    issuer = JobCertIssuer(serialize_cert(jca_cert), jca_key)
    job_a_crt, job_a_key = issuer.issue("site-1", _JOB_A, "site-1")
    job_b_crt, job_b_key = issuer.issue("site-1", _JOB_B, "site-1")
    files = {
        "rootCA.pem": serialize_cert(root_cert),
        "server.crt": serialize_cert(server_cert),
        "server.key": serialize_pri_key(server_key),
        "site-1.crt": serialize_cert(site_cert),
        "site-1.key": serialize_pri_key(site_key),
        "job_a.crt": job_a_crt,
        "job_a.key": job_a_key,
        "job_b.crt": job_b_crt,
        "job_b.key": job_b_key,
    }
    paths = {}
    for name, data in files.items():
        paths[name] = os.path.join(out_dir, name)
        with open(paths[name], "wb") as f:
            f.write(data)
    return paths


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
