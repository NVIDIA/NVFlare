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

from nvflare.client.cell.attach import make_attach_trainer_fqcn, validate_attach_profile


def test_rendezvous_fqcn_is_a_child_of_the_job_cell():
    assert make_attach_trainer_fqcn("site-1.job-1", "trainer_a") == "site-1.job-1.-client_api_trainer_a"


@pytest.mark.parametrize(
    "url",
    [
        "grpc://10.20.30.40:9000",
        "grpc://0.0.0.0:9000",
        "grpc://localhost:9000",
        "grpc://127.0.0.1:9000",
        "grpc://[::1]:9000",
    ],
)
def test_clear_profile_syntax_is_accepted_without_making_a_cj_policy_decision(url):
    assert validate_attach_profile(url, "clear") == "clear"


@pytest.mark.parametrize(
    "url",
    [
        "grpc://10.20.30.40:9000",
        "grpc://localhost:9000",
        "http://127.0.0.1:9000",
        "tcp://[::1]:9000",
    ],
)
def test_clear_network_profile_requires_explicit_security_acknowledgement(url):
    with pytest.raises(ValueError, match="explicitly set connection_security='clear'"):
        validate_attach_profile(url, None)


def test_shared_file_profile_has_no_tls_mode():
    assert validate_attach_profile("shared-file://0/var/run/nvflare", None) == "clear"


@pytest.mark.parametrize(
    "url",
    [
        "shared-file://host/var/run/nvflare",
        "shared-file:///var/run/nvflare",
        "shared-file://0",
        "shared-file:relative/path",
    ],
)
def test_invalid_shared_file_profile_is_rejected(url):
    with pytest.raises(ValueError, match="shared-file"):
        validate_attach_profile(url, "clear")


def test_shared_file_profile_rejects_tls_label():
    with pytest.raises(ValueError, match="supports only"):
        validate_attach_profile("shared-file://0/var/run/nvflare", "mtls")


def test_file_url_is_not_treated_as_the_shared_file_driver():
    with pytest.raises(ValueError, match="not an F3 transport"):
        validate_attach_profile("file:///var/run/nvflare", "clear")


def test_tls_network_route_is_accepted():
    assert validate_attach_profile("grpcs://host.example:9000", "mtls") == "mtls"


def test_secure_network_scheme_defaults_to_mtls():
    assert validate_attach_profile("grpcs://host.example:9000", None) == "mtls"


def test_bare_ca_tls_network_route_is_rejected():
    with pytest.raises(ValueError, match="bare-CA TLS attach is not supported"):
        validate_attach_profile("grpcs://host.example:9000", "tls")
