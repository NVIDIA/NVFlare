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

from unittest.mock import patch

from nvflare.fuel.hci.reg import CommandEntry
from nvflare.fuel.hci.server.authz import AuthzFilter, PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.authz import AuthorizationService, Person


class TestAuthzFilterSubmitterRole:
    """Verify that AuthzFilter uses SUBMITTER_ROLE (not SUBMITTER_ORG) for the role field."""

    def test_submitter_role_uses_correct_prop(self):
        cmd_entry = CommandEntry(
            scope="test",
            name="check_status",
            desc="test",
            usage="test",
            handler=lambda *a: None,
            authz_func=lambda conn, args: PreAuthzReturnCode.REQUIRE_AUTHZ,
            visible=True,
            confirm=False,
            client_cmd=False,
        )

        class MockConn:
            def __init__(self):
                self._p = {}
                self._p[ConnProps.CMD_ENTRY] = cmd_entry
                self._p[ConnProps.USER_NAME] = "user"
                self._p[ConnProps.USER_ORG] = "user_org"
                self._p[ConnProps.USER_ROLE] = "user_role"
                self._p[ConnProps.SUBMITTER_NAME] = "submitter"
                self._p[ConnProps.SUBMITTER_ORG] = "test_org"
                self._p[ConnProps.SUBMITTER_ROLE] = "test_role"

            def get_prop(self, k, d=""):
                return self._p.get(k, d)

            def append_error(self, *a, **kw):
                pass

        captured = {}
        orig_init = Person.__init__

        def patched_init(self, name="", org="", role=""):
            orig_init(self, name=name, org=org, role=role)
            if name == "submitter":
                captured["role"] = role

        with patch.object(Person, "__init__", patched_init), patch.object(
            AuthorizationService, "authorize", staticmethod(lambda ctx: (True, ""))
        ):

        assert captured.get("role") == "test_role", (
            f"Expected role='test_role' but got role='{captured.get('role')}'. "
            f"AuthzFilter may be using SUBMITTER_ORG instead of SUBMITTER_ROLE."
        )
