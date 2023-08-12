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

import io
import json
import os
from typing import Optional
from zipfile import ZipFile

from nvflare.apis.fl_constant import JobConstants
from nvflare.apis.job_def import ALL_SITES, JobMetaKey
from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.fuel.utils.zip_utils import normpath_for_zip, zip_directory_to_bytes


def _get_default_meta(job_folder_name: str) -> str:
    # A format string for the dummy meta.json
    meta = f"""{{
                 "{JobMetaKey.JOB_NAME.value}": "{job_folder_name}",
                 "{JobMetaKey.JOB_FOLDER_NAME.value}": "{job_folder_name}",
                 "{JobMetaKey.RESOURCE_SPEC.value}": {{ }},
                 "{JobMetaKey.DEPLOY_MAP.value}": {{ "{job_folder_name}": ["{ALL_SITES}"] }},
                 "{JobMetaKey.MIN_CLIENTS.value}": 1
               }}
            """
    return meta


def convert_legacy_zipped_app_to_job(zip_data: bytes) -> bytes:
    """Convert a legacy app in zip into job layout in memory.

    Args:
        zip_data: The input zip data

    Returns:
        The converted zip data
    """

    meta: Optional[dict] = None
    reader = io.BytesIO(zip_data)
    with ZipFile(reader, "r") as in_zip:
        info_list = in_zip.infolist()
        folder_name = info_list[0].filename.split("/")[0]
        meta_file = os.path.join(folder_name, JobConstants.META)
        meta_json = normpath_for_zip(os.path.join(folder_name, JobConstants.META_FILE))
        meta_path = None
        for ext, fmt in ConfigFormat.config_ext_formats().items():
            meta_file_path = normpath_for_zip(f"{meta_file}{ext}")
            if next((info for info in info_list if info.filename == meta_file_path), None):
                # Already in job layout
                meta_path = meta_file_path
                config_loader = ConfigFactory.get_config_loader(fmt)
                meta_data = in_zip.read(meta_path)
                meta = config_loader.load_config_from_str(meta_data.decode()).to_dict()
                if JobMetaKey.JOB_FOLDER_NAME.value not in meta:
                    meta[JobMetaKey.JOB_FOLDER_NAME.value] = folder_name
                else:
                    return zip_data
                break

        writer = io.BytesIO()
        with ZipFile(writer, "w") as out_zip:
            if meta:
                out_zip.writestr(meta_json, json.dumps(meta))
                out_zip.comment = in_zip.comment  # preserve the comment
                for info in info_list:
                    if info.filename != meta_path:
                        out_zip.writestr(info, in_zip.read(info.filename))
            else:
                out_zip.writestr(meta_json, _get_default_meta(folder_name))
                # Push everything else to a sub folder with the same name:
                # hello-pt/README.md -> hello-pt/hello-pt/README.md
                for info in info_list:
                    name = info.filename
                    content = in_zip.read(name)
                    path = folder_name + "/" + name
                    info.filename = path
                    out_zip.writestr(info, content)

        return writer.getvalue()


def load_job_def_bytes(from_path: str, def_name: str) -> bytes:
    """Load a job definition from specified path and return zipped bytes

    Args:
        from_path: path where the job definition is located
        def_name: name of the job

    Returns:

    """
    # zip the job folder
    data = zip_directory_to_bytes(from_path, def_name)
    return convert_legacy_zipped_app_to_job(data)
