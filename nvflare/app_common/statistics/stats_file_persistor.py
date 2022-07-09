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
import json
import os
from typing import List

from nvflare.apis.fl_context import FLContext
from nvflare.apis.stats_persistence import StatsWriter, FileFormat
from nvflare.apis.storage import StorageException
from nvflare.app_common.statistics.stats_def import (
    DatasetStatistics,
)
from nvflare.app_common.storages.filesystem_storage import _write
from nvflare.apis.fl_constant import FLContextKey


class StatsFileWriter(StatsWriter):
    def __init__(self, root_dir: str):
        super().__init__()
        """Creates a StatsFileWriter.
        Args:
            root_dir: where to store the states.
        """
        if len(root_dir) == 0:
            raise ValueError(f"root_dir {root_dir} is empty string")

        self.root_dir = root_dir

    def save(self,
             relative_path: str,
             data: List[DatasetStatistics],
             file_format: FileFormat,
             overwrite_existing,
             fl_ctx: FLContext):

        root_dir = self.get_output_path(relative_path, fl_ctx)
        self.validate_directory(root_dir)

        full_uri = os.path.join(root_dir, relative_path.lstrip(" /"))
        data_exists = os.path.isfile(full_uri)
        if data_exists and not overwrite_existing:
            raise StorageException("object {} already exists and overwrite_existing is False".format(full_uri))

        if file_format == FileFormat.JSON:
            content = json.dumps(data)
        else:
            raise NotImplemented

        self.log_info(fl_ctx, f"save data to {full_uri} in {file_format} format")
        _write(full_uri, content)

    def get_output_path(self, relative_path: str, fl_ctx: FLContext) -> str:
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        workspace_dir = workspace.get_root_dir()
        # todo: we may need to use parent directory of workspace as value, as the workspace for server has /server in it
        return self.root_dir.replace("{workspace_dir}", workspace_dir)

    def validate_directory(self, full_path: str):
        if not os.path.isabs(full_path):
            raise ValueError(f"root_dir {full_path} must be an absolute path.")
        if os.path.exists(full_path) and not os.path.isdir(full_path):
            raise ValueError(f"root_dir {full_path} exists but is not a directory.")
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=False)
