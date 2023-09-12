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
import uuid
from enum import Enum
from typing import Any, Dict, Union

TEN_MEGA = 10 * 1024 * 1024


class DatumType(Enum):
    BLOB = 1
    FILE = 2


class Datum:
    """Datum is a class that holds information for externalized data"""

    def __init__(self, datum_type: DatumType, value: Any):
        self.datum_id = str(uuid.uuid4())
        self.datum_type = datum_type
        self.value = value

    @staticmethod
    def blob_datum(blob: Union[bytes, bytearray, memoryview]):
        """Factory method to create a BLOB datum"""
        return Datum(DatumType.BLOB, blob)

    @staticmethod
    def file_datum(path: str):
        """Factory method to crate a file datum"""
        return Datum(DatumType.FILE, path)


class DatumRef:
    """A reference to externalized datum. If unwrap is true, the reference will be removed and replaced with the
    content of the datum"""

    def __init__(self, datum_id: str, unwrap=False):
        self.datum_id = datum_id
        self.unwrap = unwrap


class DatumManager:
    def __init__(self, threshold=TEN_MEGA):
        self.threshold = threshold
        self.datums: Dict[str, Datum] = {}

    def get_datums(self):
        return self.datums

    def externalize(self, data: Any):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            return data

        if len(data) >= self.threshold:
            # turn it to Datum
            d = Datum.blob_datum(data)
            self.datums[d.datum_id] = d
            return DatumRef(d.datum_id, True)
        else:
            return data

    def internalize(self, data: Any) -> Any:
        if not isinstance(data, DatumRef):
            return data

        d = self.datums.get(data.datum_id)
        if not d:
            raise ValueError(f"can't find datum for {data.datum_id}")

        if data.unwrap:
            return d.value
        else:
            return d
