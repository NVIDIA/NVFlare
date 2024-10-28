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
from typing import List

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import StorageException
from nvflare.app_common.psi.psi_spec import PSIWriter


def _validate_directory(full_path: str):
    if not os.path.isabs(full_path):
        raise ValueError(f"path {full_path} must be an absolute path.")
    parent_dir = os.path.dirname(full_path)
    if os.path.exists(parent_dir) and not os.path.isdir(parent_dir):
        raise ValueError(f"directory {parent_dir} exists but is not a directory.")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=False)


class FilePSIWriter(PSIWriter):
    def __init__(self, output_path: str):
        super().__init__()
        if len(output_path) == 0:
            raise ValueError(f"output_path {output_path} is empty string")
        self.output_path = output_path

    def save(
        self,
        intersection: List[str],
        overwrite_existing,
        fl_ctx: FLContext,
    ):

        full_uri = self.get_output_path(fl_ctx)
        _validate_directory(full_uri)

        data_exists = os.path.isfile(full_uri)
        if data_exists and not overwrite_existing:
            raise StorageException("object {} already exists and overwrite_existing is False".format(full_uri))
        self.log_info(fl_ctx, f"trying to save data to {full_uri}")

        with open(full_uri, "w") as fp:
            fp.write("\n".join(intersection))

        self.log_info(fl_ctx, f"file {full_uri} saved")

    def get_output_path(self, fl_ctx: FLContext) -> str:
        job_dir = os.path.dirname(os.path.abspath(fl_ctx.get_prop(FLContextKey.APP_ROOT)))
        self.log_info(fl_ctx, "job dir = " + job_dir)
        return os.path.join(job_dir, fl_ctx.get_identity_name(), self.output_path)
