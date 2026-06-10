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

import pytest

from nvflare.private.fed.server.fed_server import build_root_urls


def test_shared_port_uses_single_admin_listener():
    assert build_root_urls("grpc", 8002, 8002) == ["grpc://0:8002?admin_listener=true"]


def test_dedicated_admin_port_adds_admin_listener():
    assert build_root_urls("grpc", 8002, 8003) == [
        "grpc://0:8002",
        "grpc://0:8003?admin_listener=true",
    ]


def test_admin_conn_security_override_applies_to_admin_listener_only():
    # OIDC admin consoles need one-way TLS; the FL listener URL carries no override,
    # so it keeps the cell-level (mTLS) credentials, and the relaxed listener is
    # restricted to admin endpoints.
    assert build_root_urls("grpc", 8002, 8003, "tls") == [
        "grpc://0:8002",
        "grpc://0:8003?admin_listener=true&connection_security=tls&admin_only=true",
    ]


def test_admin_conn_security_override_requires_dedicated_admin_port():
    with pytest.raises(RuntimeError, match="dedicated admin port"):
        build_root_urls("grpc", 8002, 8002, "tls")
