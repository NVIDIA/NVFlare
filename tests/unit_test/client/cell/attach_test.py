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

from nvflare.client.cell.attach import make_attach_trainer_fqcn, validate_attach_transport


def test_rendezvous_fqcn_matches_ipc_site_child_shape():
    assert make_attach_trainer_fqcn("site-1", "trainer_a") == "site-1.-client_api_trainer_a"


@pytest.mark.parametrize(
    "url",
    [
        "grpc://10.20.30.40:9000",
        "grpc://0.0.0.0:9000",
        "grpc://localhost:9000",
    ],
)
def test_clear_non_literal_loopback_routes_are_rejected(url):
    with pytest.raises(ValueError, match="cleartext non-loopback"):
        validate_attach_transport(url, "clear", allow_insecure_attach=False)


def test_literal_loopback_and_shared_file_do_not_need_insecure_opt_in():
    assert validate_attach_transport("grpc://127.0.0.1:9000", "clear", False) == "clear"
    assert validate_attach_transport("grpc://[::1]:9000", "clear", False) == "clear"
    assert validate_attach_transport("shared-file://0/var/run/nvflare", None, False) == "clear"


def test_tls_network_route_is_accepted():
    assert validate_attach_transport("grpcs://host.example:9000", "mtls", False) == "mtls"


def test_bare_ca_tls_network_route_is_rejected():
    with pytest.raises(ValueError, match="bare-CA TLS attach is not supported"):
        validate_attach_transport("grpcs://host.example:9000", "tls", False)
