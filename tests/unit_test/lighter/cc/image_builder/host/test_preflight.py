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

"""QGS configurations supplied by Intel packages and sectioned variants."""

import tempfile
import unittest
from pathlib import Path

from cvm.common.errors import BuildError
from cvm.host.preflight import check_qgs_config


class PreflightTests(unittest.TestCase):
    def test_headerless_and_sectioned_qgs_configs(self):
        for content in (
            "# Intel QGS\nport = 4050\nnumber_threads = 4\n",
            "[server]\nport = 4050 # vsock\n",
            "[DEFAULT]\nport = 4050\n",
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "qgs.conf"
                path.write_text(content)
                check_qgs_config(path)

    def test_missing_malformed_commented_and_wrong_ports_fail_closed(self):
        for content in (None, "", "# port = 4050\n", "port=4051\n", "port=4050\nport=4051\n", "[broken\nport=4050"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "qgs.conf"
                if content is not None:
                    path.write_text(content)
                with self.assertRaises(BuildError):
                    check_qgs_config(path)
