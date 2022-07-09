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

from abc import ABC, abstractmethod
from enum import Enum

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class FileFormat(Enum):
    JSON = 0
    CSV = 1


class StatsWriter(FLComponent, ABC):
    @abstractmethod
    def save(self,
             relative_path: str,
             data: bytes,
             file_format: FileFormat,
             overwrite_existing: bool,
             fl_ctx: FLContext):
        """
            save data to path in given file_format. The implementor needs to make sure the
            data is serializable to the specified file_format.
        :param relative_path: storage relative path to the root_uri, if the root URI is s3://bucket/stats/ the relative_path
               could be "statsjson" and the full path be s3://bucket/stats/stats.json
        :param data: the data to be saved
        :param file_format: File format such as JSON
        :param overwrite_existing
        :param fl_ctx: FLContext
        :return: None
        """
        pass
