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

import uuid

import pytest

from nvflare.fuel.hci.security import get_certificate_common_name, hash_password, make_session_token, verify_password


class TestSecurityUtils:
    @pytest.mark.parametrize("password_to_hash", ["abcde", "xyz"])
    def test_hash_password(self, password_to_hash):
        hashed_password = hash_password(password_to_hash)
        assert verify_password(hashed_password, password_to_hash)

    @pytest.mark.parametrize(
        "password_hash, password_to_verify, expected_result",
        [
            [
                "b1a020416bc1479da3f937af46f90a09a4c09cd6271609105f80e7c9e7fd461ffe463784834ea63c7525f85b80435bbc0dfba614570f23aaccbd115bbef81a57b2e73a39563f0d1b75132c8b9e1b53dc94a1525be01d0e6862e577360e820592",
                "abcde",
                False,
            ],
            [
                "b1a020416bc1479da3f937af46f90a09a4c09cd6271609105f80e7c9e7fd461ffe463784834ea63c7525f85b80435bbc0dfba614570f23aaccbd115bbef81a57b2e73a39563f0d1b75132c8b9e1b53dc94a1525be01d0e6862e577360e820592",
                "xyz",
                True,
            ],
        ],
    )
    def test_verify_password(self, password_hash, password_to_verify, expected_result):
        result = verify_password(password_hash, password_to_verify)
        assert result == expected_result

    def test_make_session_token(self):
        uuid.UUID(make_session_token())
        assert True

    @pytest.mark.parametrize(
        "cert, expected",
        [
            (None, None),
            ({}, None),
            ({"subject": {}}, None),
            (
                {
                    "subject": (
                        (("description", "571208-SLe257oHY9fVQ07Z"),),
                        (("countryName", "US"),),
                        (("stateOrProvinceName", "California"),),
                        (("localityName", "San Francisco"),),
                        (("organizationName", "Electronic Frontier Foundation, Inc."),),
                        (("commonName", "*.eff.org"),),
                        (("emailAddress", "hostmaster@eff.org"),),
                    )
                },
                "*.eff.org",
            ),
        ],
    )
    def test_get_certificate_common_name(self, cert, expected):
        result = get_certificate_common_name(cert)
        assert result == expected
