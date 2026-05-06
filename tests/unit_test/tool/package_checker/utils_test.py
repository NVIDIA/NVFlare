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

from nvflare.tool.package_checker.utils import try_bind_address, try_write_dir


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
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            assert try_bind_address(host="localhost", port=12345) is None

    def test_try_bind_address_error(self):
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.bind.side_effect = OSError("Test")
            mock_socket_cls.return_value = mock_sock
            assert try_bind_address(host="localhost", port=12345).args == OSError("Test").args
