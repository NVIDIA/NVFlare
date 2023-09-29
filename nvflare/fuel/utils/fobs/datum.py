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
    TEXT = 1  # for text string
    BLOB = 2  # for binary bytes
    FILE = 3  # for file name


class Datum:
    """Datum is a class that holds information for externalized data"""

    def __init__(self, datum_type: DatumType, value: Any):
        """Constructor of Datum object

        Args:
            datum_type: type of the datum.
            value: value of the datum

        """
        self.datum_id = str(uuid.uuid4())
        self.datum_type = datum_type
        self.value = value
        self.restore_func = None  # func to restore original object.
        self.restore_func_data = None  # arg to the restore func

    def set_restore_func(self, func, func_data):
        """Set the restore function and func data.
        Restore func is set during the serialization process. If set, the func will be called after the serialization
        to restore the serialized object back to its original state.

         Args:
             func: the restore function
             func_data: arg passed to the restore func when called

         Returns: None

        """
        if not callable(func):
            raise ValueError(f"func must be callable but got {type(func)}")
        self.restore_func = func
        self.restore_func_data = func_data

    @staticmethod
    def blob_datum(blob: Union[bytes, bytearray, memoryview]):
        """Factory method to create a BLOB datum"""
        return Datum(DatumType.BLOB, blob)

    @staticmethod
    def text_datum(text: str):
        """Factory method to create a TEXT datum"""
        return Datum(DatumType.TEXT, text)

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
    def __init__(self, threshold=None):
        if not threshold:
            threshold = TEN_MEGA

        if not isinstance(threshold, int):
            raise TypeError(f"threshold must be int but got {type(threshold)}")

        if threshold <= 0:
            raise ValueError(f"threshold must > 0 but got {threshold}")

        self.threshold = threshold
        self.datums: Dict[str, Datum] = {}

        # some decomposers (e.g. Shareable, Learnable, etc.) make a shallow copy of the original object before
        # serialization. After serialization, only the values in the copy are restored. We need to keep a ref
        # from the copy to the original object so that values in the original are also restored.
        self.obj_copies = {}  # copy id => original object

    def register_copy(self, obj_copy, original_obj):
        """Register the object_copy => original object

        Args:
            obj_copy: a copy of the original object
            original_obj: the original object

        Returns: None

        """
        self.obj_copies[id(obj_copy)] = original_obj

    def get_original(self, obj_copy) -> Any:
        """Get the registered original object from the object copy.

        Args:
            obj_copy: a copy of the original object

        Returns: the original object if found; None otherwise.

        """
        return self.obj_copies.get(id(obj_copy))

    def get_datums(self):
        return self.datums

    def get_datum(self, datum_id: str):
        return self.datums.get(datum_id)

    def externalize(self, data: Any):
        if not isinstance(data, (bytes, bytearray, memoryview, Datum, str)):
            return data

        if isinstance(data, Datum):
            # this is an app-defined datum. we need to keep it as is when deserialized.
            # hence unwrap is set to False in the DatumRef.
            self.datums[data.datum_id] = data
            return DatumRef(data.datum_id, False)

        if len(data) >= self.threshold:
            # turn it to Datum
            if isinstance(data, str):
                d = Datum.text_datum(data)
            else:
                d = Datum.blob_datum(data)
            self.datums[d.datum_id] = d
            return DatumRef(d.datum_id, True)
        else:
            return data

    def internalize(self, data: Any) -> Any:
        if not isinstance(data, DatumRef):
            return data

        d = self.get_datum(data.datum_id)
        if not d:
            raise RuntimeError(f"can't find datum for {data.datum_id}")

        if d.datum_type == DatumType.FILE:
            return d
        elif data.unwrap:
            return d.value
        else:
            return d
