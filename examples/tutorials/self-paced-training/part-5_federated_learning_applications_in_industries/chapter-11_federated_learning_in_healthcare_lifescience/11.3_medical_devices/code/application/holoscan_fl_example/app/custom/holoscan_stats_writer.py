# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from datetime import datetime

from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import StorageException
from nvflare.app_common.abstract.statistics_writer import StatisticsWriter
from nvflare.app_common.utils.json_utils import ObjectEncoder
from nvflare.fuel.utils.class_loader import load_class


class HoloscanStatsWriter(StatisticsWriter):
    def __init__(self, output_path: str, output_file_name: str, json_encoder_path: str = ""):
        super().__init__()
        self.job_dir = None
        if len(output_path) == 0:
            raise ValueError(f"output_path {output_path} is empty string")

        self.output_path = output_path
        self.output_file_name = output_file_name
        if json_encoder_path == "":
            self.json_encoder_class = ObjectEncoder
        else:
            self.json_encoder_path = json_encoder_path
            self.json_encoder_class = load_class(json_encoder_path)

    def save(
        self,
        data: dict,
        overwrite_existing,
        fl_ctx: FLContext,
    ):

        full_uri = self.get_output_path(fl_ctx)
        self._validate_directory(full_uri)

        data_exists = os.path.isfile(full_uri)
        if data_exists and not overwrite_existing:
            raise StorageException("object {} already exists and overwrite_existing is False".format(full_uri))

        content = json.dumps(data, cls=self.json_encoder_class)

        self.log_info(fl_ctx, f"trying to save data to {full_uri}")
        with open(full_uri, "w") as outfile:
            outfile.write(content)

        self.log_info(fl_ctx, f"file {full_uri} saved")

    def get_output_path(self, fl_ctx: FLContext) -> str:
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = current_timestamp
        path = os.path.join(self.output_path, directory_name)
        os.makedirs(path, exist_ok=True)
        print(f"Output directory created: {path}")
        return os.path.join(path, self.output_file_name)

    def _validate_directory(self, full_path: str):
        if not os.path.isabs(full_path):
            raise ValueError(f"path {full_path} must be an absolute path.")
        parent_dir = os.path.dirname(full_path)
        if os.path.exists(parent_dir) and not os.path.isdir(parent_dir):
            raise ValueError(f"directory {parent_dir} exists but is not a directory.")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=False)
