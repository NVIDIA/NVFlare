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


import pickle


class ComponentState(dict):
    def __init__(self):
        super().__init__()

    def set_prop(self, key: str, value):
        """method to add the component state data into the key value pairs.

        Args:
            key: state data key
            value: state value

        Returns: N/A

        """
        self[key] = value

    def get_prop(self, key: str, default=None):
        """method to retrievee the component state data value.

        Args:
            key: key
            default: default value

        Returns: value data

        """
        return self.get(key, default)

    def to_bytes(self) -> bytes:
        """method to serialize the Persistable object into bytes.

        Returns:
            object serialized in bytes.

        """
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes):
        """method to convert the object bytes into Persistable object.

        Args:
            data: a bytes object

        Returns:
            an object loaded by pickle from data

        """
        return pickle.loads(data)
