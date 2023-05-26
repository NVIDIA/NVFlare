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

from typing import Any


class FileAccessor:
    def write(self, data: Any, file_path: str):
        """Write the specified data to file(s) in the specified path

        Args:
            data: data to be written
            file_path: where the data is to be written

        Returns:

        """
        pass

    def read(self, file_path: str) -> Any:
        """Read the data located at the specified file_path

        Args:
            file_path: location of the data to be read

        Returns: the data object read

        """
        pass
