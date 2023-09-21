# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
import tempfile
import uuid

from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.fuel.utils import fobs


class TestLobs:
    @classmethod
    def setup_class(cls):
        flare_decomposers.register()

    def test_handle_bytes(self):
        flare_decomposers.register()
        d = Shareable()
        d["x"] = os.urandom(200)
        d["y"] = b"123456789012345678901234567890123456789012345678901234567890"
        d["z"] = {
            "za": 12345,
            "zb": b"123456789012345678901234567890123456789012345678",
        }
        ds = fobs.dumps(d, max_value_size=15)
        dd = fobs.loads(ds)
        assert d == dd

    def test_handle_file(self):
        flare_decomposers.register()
        d = Shareable()
        d["x"] = os.urandom(200)
        d["y"] = b"123456789012345678901234567890123456789012345678901234567890"
        d["z"] = {
            "za": 12345,
            "zb": b"123456789012345678901234567890123456789012345678",
        }

        with tempfile.TemporaryDirectory() as td:
            file_path = os.path.join(td, str(uuid.uuid4()))
            fobs.dumpf(d, file_path, max_value_size=15)
            df = fobs.loadf(file_path)
            assert df == d
