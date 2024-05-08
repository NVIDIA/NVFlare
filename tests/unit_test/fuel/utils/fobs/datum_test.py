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

from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager

BLOB_SIZE = 1024 * 1024  # 1M


class TestDatum:

    test_data = Shareable()
    test_data["data"] = {
        "key1": "Test",
        "blob1": bytes(BLOB_SIZE),
        "member": {"key2": 123, "blob2": bytearray(BLOB_SIZE)},
    }

    def test_datum(self):
        flare_decomposers.register()
        manager = DatumManager(BLOB_SIZE)
        buf = fobs.serialize(TestDatum.test_data, manager)
        assert len(buf) < BLOB_SIZE
        datums = manager.get_datums()
        assert len(datums) == 2

        data = fobs.deserialize(buf, manager)
        assert isinstance(data["data"]["blob1"], bytes)
