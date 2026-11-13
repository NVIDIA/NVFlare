# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import StorageException
from nvflare.app_common.abstract.statistics_writer import StatisticsWriter
from nvflare.app_common.utils.json_utils import ObjectEncoder
from nvflare.fuel.utils.class_loader import load_class

OBJECT_ENCODER_PATH = "nvflare.app_common.utils.json_utils.ObjectEncoder"
_BUILT_IN_JSON_ENCODER_PATHS = {"", OBJECT_ENCODER_PATH}


class JsonStatsFileWriter(StatisticsWriter):
    def __init__(self, output_path: str, json_encoder_path: str = ""):
        super().__init__()
        self.job_dir = None
        self.json_encoder_class = None
        if len(output_path) == 0:
            raise ValueError(f"output_path {output_path} is empty string")
        if not isinstance(json_encoder_path, str):
            raise TypeError(f"json_encoder_path must be str but got {type(json_encoder_path)}")

        self.output_path = output_path
        self.json_encoder_path = json_encoder_path

    def save(
        self,
        data: dict,
        overwrite_existing,
        fl_ctx: FLContext,
    ):

        full_uri = self.get_output_path(fl_ctx)
        content = json.dumps(data, cls=self._get_json_encoder_class(fl_ctx))
        self._validate_directory(full_uri)

        self.log_info(fl_ctx, f"trying to save data to {full_uri}")
        with self._open_output_file(full_uri, overwrite_existing) as outfile:
            outfile.write(content)

        self.log_info(fl_ctx, f"file {full_uri} saved")

    def get_output_path(self, fl_ctx: FLContext) -> str:
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        if not app_root:
            raise ValueError(f"missing {FLContextKey.APP_ROOT} in fl_ctx")

        self.job_dir = os.path.realpath(os.path.dirname(os.path.abspath(app_root)))
        self.log_info(fl_ctx, "job dir = " + self.job_dir)
        return self._resolve_output_path(self.job_dir)

    def _validate_directory(self, full_path: str):
        if not os.path.isabs(full_path):
            raise ValueError(f"path {full_path} must be an absolute path.")
        parent_dir = os.path.dirname(full_path)
        if os.path.exists(parent_dir) and not os.path.isdir(parent_dir):
            raise ValueError(f"directory {parent_dir} exists but is not a directory.")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, mode=0o700, exist_ok=False)

    @staticmethod
    def _open_output_file(full_path: str, overwrite_existing: bool):
        flags = os.O_WRONLY | os.O_CREAT
        if overwrite_existing:
            flags |= os.O_TRUNC
        else:
            flags |= os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            # POSIX-only defense in depth; it protects the final path component.
            # Existing intermediate symlink escapes are rejected by realpath/commonpath before this open.
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(full_path, flags, 0o600)
        except FileExistsError as ex:
            raise StorageException(f"object {full_path} already exists and overwrite_existing is False") from ex
        return os.fdopen(fd, "w", encoding="utf-8")

    def _resolve_output_path(self, job_dir: str) -> str:
        if os.path.isabs(self.output_path):
            raise ValueError(f"output_path {self.output_path} must be relative to the job directory.")

        full_path = os.path.realpath(os.path.join(job_dir, self.output_path))
        if os.path.commonpath([job_dir, full_path]) != job_dir:
            raise ValueError(f"output_path {self.output_path} must stay inside the job directory.")

        return full_path

    def _get_json_encoder_class(self, fl_ctx: FLContext):
        if self.json_encoder_path in _BUILT_IN_JSON_ENCODER_PATHS:
            return ObjectEncoder

        if not self._job_has_byoc(fl_ctx):
            raise ValueError("custom json_encoder_path is only allowed for BYOC jobs.")

        if self.json_encoder_class is None:
            json_encoder_class = load_class(self.json_encoder_path)
            if not issubclass(json_encoder_class, json.JSONEncoder):
                raise TypeError(f"json_encoder_path {self.json_encoder_path} must be a JSONEncoder class.")
            self.json_encoder_class = json_encoder_class

        return self.json_encoder_class

    @staticmethod
    def _job_has_byoc(fl_ctx: FLContext) -> bool:
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not isinstance(job_meta, dict):
            return False
        return bool(job_meta.get(AppValidationKey.BYOC))
