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

import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from requests import Response

from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.tool.package_checker.utils import (
    NVFlareRole,
    check_overseer_running,
    check_response,
    get_required_args_for_overseer_agent,
    try_bind_address,
    try_write_dir,
)


def _mock_response(code) -> Response:
    resp = MagicMock(spec=Response)
    resp.json.return_value = {}
    resp.status_code = code
    return resp


class TestUtils:
    def test_try_write_exist(self):
        tempdir = tempfile.mkdtemp()
        assert try_write_dir(tempdir) is None
        shutil.rmtree(tempdir)

    def test_try_write_non_exist(self):
        tempdir = tempfile.mkdtemp()
        shutil.rmtree(tempdir)
        assert try_write_dir(tempdir) is None

    def test_try_write_exception(self):
        with patch("os.path.exists", side_effect=OSError("Test")):
            assert try_write_dir("hello").args == OSError("Test").args

    def test_try_bind_address(self):
        assert try_bind_address(host="localhost", port=get_open_ports(1)[0]) is None

    def test_try_bind_address_error(self):
        host = "localhost"
        port = get_open_ports(1)[0]
        with patch("socket.socket.bind", side_effect=OSError("Test")):
            assert try_bind_address(host=host, port=port).args == OSError("Test").args

    @pytest.mark.parametrize("resp, result", [(None, False), (_mock_response(200), True), (_mock_response(404), False)])
    def test_check_response(self, resp, result):
        assert check_response(resp=resp) == result

    def test_check_overseer_running(self):
        with patch("nvflare.tool.package_checker.utils._create_http_session") as mock2:
            mock2.return_value = None
            with patch("nvflare.tool.package_checker.utils._prepare_data") as mock3:
                mock3.return_value = None
                with patch("nvflare.tool.package_checker.utils._send_request") as mock4:
                    mock4.return_value = _mock_response(200)
                    resp, err = check_overseer_running(
                        startup="test",
                        overseer_agent_args={"overseer_end_point": "random"},
                        role="",
                    )
                    assert resp.status_code == 200

    @pytest.mark.parametrize(
        "overseer_agent_class, role, result",
        [
            (
                "nvflare.ha.overseer_agent.HttpOverseerAgent",
                NVFlareRole.SERVER,
                ["overseer_end_point", "role", "project", "name", "fl_port", "admin_port"],
            ),
            (
                "nvflare.ha.overseer_agent.HttpOverseerAgent",
                NVFlareRole.CLIENT,
                ["overseer_end_point", "role", "project", "name"],
            ),
            (
                "nvflare.ha.overseer_agent.HttpOverseerAgent",
                NVFlareRole.ADMIN,
                ["overseer_end_point", "role", "project", "name"],
            ),
            ("nvflare.ha.dummy_overseer_agent.DummyOverseerAgent", NVFlareRole.SERVER, ["sp_end_point"]),
            ("nvflare.ha.dummy_overseer_agent.DummyOverseerAgent", NVFlareRole.CLIENT, ["sp_end_point"]),
            ("nvflare.ha.dummy_overseer_agent.DummyOverseerAgent", NVFlareRole.ADMIN, ["sp_end_point"]),
        ],
    )
    def test_get_required_args_for_overseer_agent(self, overseer_agent_class, role, result):
        assert get_required_args_for_overseer_agent(overseer_agent_class, role) == result
