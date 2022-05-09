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

import pytest

from nvflare.lighter.spec import Participant


class TestParticipant:
    @pytest.mark.parametrize(
        "type,invalid_name",
        [
            ("server", "server_"),
            ("server", "server@"),
            ("server", "-server"),
            ("client", "client!"),
            ("client", "client@"),
            ("admin", "admin"),
            ("admin", "admin@example_1.com"),
            ("overseer", "overseer_"),
        ],
    )
    def test_invalid_name(self, type, invalid_name):
        with pytest.raises(ValueError):
            _ = Participant(name=invalid_name, org="org", type=type)

    @pytest.mark.parametrize(
        "invalid_org",
        [("org-"), ("org@"), ("org!"), ("org~")],
    )
    def test_invalid_org(self, invalid_org):
        with pytest.raises(ValueError):
            _ = Participant(name="server", type="server", org=invalid_org)

    @pytest.mark.parametrize(
        "invalid_type",
        [("type1"), ("type@"), ("type!"), ("type~"), ("gateway"), ("firewall")],
    )
    def test_invalid_type(self, invalid_type):
        with pytest.raises(ValueError):
            _ = Participant(name="server", type=invalid_type, org="org")
