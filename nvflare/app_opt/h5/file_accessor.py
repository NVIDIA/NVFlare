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

import h5py

from nvflare.fuel.utils.pipe.file_accessor import FileAccessor

DEFAULT_KEY = "__default__"


class H5FileAccessor(FileAccessor):
    def write(self, data, file_path: str):
        with h5py.File(file_path, "w") as file:
            if not isinstance(data, dict):
                file.create_dataset(DEFAULT_KEY, data=data)
            else:
                for key, value in data.items():
                    file.create_dataset(key, data=value)

    def read(self, file_path: str):
        with h5py.File(file_path, "r") as file:
            if len(file.keys()) == 1 and list(file.keys())[0] == DEFAULT_KEY:
                data = file[DEFAULT_KEY][()]
            else:
                data = {}
                for key in file.keys():
                    data[key] = file[key][()]
        return data
