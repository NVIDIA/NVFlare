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
from typing import Optional

from nvflare.app_common.dex.dxo_exchanger import DXOExchanger
from nvflare.fuel.utils.pipe.file_accessor import FileAccessor
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pickle_file_accessor import PickleFileAccessor


class DXOFileExchanger(DXOExchanger):
    def create_pipe(self, data_exchange_path: str, file_accessor: Optional[FileAccessor] = None) -> FilePipe:
        data_exchange_path = os.path.abspath(data_exchange_path)
        file_pipe = FilePipe(data_exchange_path)
        if file_accessor is not None:
            file_pipe.set_file_accessor(file_accessor)
        else:
            file_pipe.set_file_accessor(PickleFileAccessor())

        return file_pipe
