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

import io
import json
import os
from zipfile import ZipFile

from nvflare.apis.job_def import ALL_SITES, JobMetaKey

META_NAME = "meta.json"
# A format string for the dummy meta.json


def _get_default_meta(job_folder_name: str) -> str:
    meta = f"""{{
                 "{JobMetaKey.JOB_FOLDER_NAME.value}": "{job_folder_name}",
                 "{JobMetaKey.RESOURCE_SPEC.value}": {{ }},
                 "{JobMetaKey.DEPLOY_MAP}": {{ "{job_folder_name}": ["{ALL_SITES}"] }},
                 "{JobMetaKey.MIN_CLIENTS}": 1
               }}
            """
    return meta


def _path_join(base: str, *parts: str) -> str:
    path = os.path.normpath(os.path.join(base, *parts))
    path = os.path.splitdrive(path)[1]
    # ZIP spec requires forward slashes
    return path.replace("\\", "/")


def remove_leading_dotdot(path: str) -> str:
    while path.startswith("../"):
        path = path[3:]
    return path


def split_path(path: str) -> (str, str):
    """Split a path into prefix and folder name

    Args:
        path: Path to split

    Returns: A tuple of (prefix, folder_name)
    """

    if path.endswith("/"):
        full_path = path[:-1]
    else:
        full_path = path

    return os.path.split(full_path)


def get_all_file_paths(directory):
    """Get all file paths in the directory.

    Args:
        directory: directory to get all paths for

    Returns: all paths in the provided directory
    """
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_paths.append(_path_join(root, filename))
        for dir_name in directories:
            file_paths.append(_path_join(root, dir_name))

    return file_paths


def _zip_directory(root_dir: str, folder_name: str, writer: io.BytesIO):
    """Create a zip archive file for the specified directory.

    Args:
        root_dir: root path that contains the folder to be zipped
        folder_name: path to the folder to be zipped, relative to root_dir
        writer: file to write to
    """
    dir_name = _path_join(root_dir, folder_name)
    assert os.path.exists(dir_name), 'directory "{}" does not exist'.format(dir_name)
    assert os.path.isdir(dir_name), '"{}" is not a valid directory'.format(dir_name)

    file_paths = get_all_file_paths(dir_name)
    if folder_name:
        prefix_len = len(split_path(dir_name)[0]) + 1
    else:
        prefix_len = len(dir_name) + 1

    # writing files to a zipfile
    with ZipFile(writer, "w") as z:
        # writing each file one by one
        for full_path in file_paths:
            rel_path = full_path[prefix_len:]
            z.write(full_path, arcname=rel_path)


def zip_directory_to_bytes(root_dir: str, folder_name: str) -> bytes:
    bio = io.BytesIO()
    _zip_directory(root_dir, folder_name, bio)
    return bio.getvalue()


def _unzip_all(reader, output_dir_name: str):
    """Decompress a zip archive file and extract all files to the specified output directory.

    Args:
        reader: the input zip reader
        output_dir_name: the output directory for extracted content
    """
    assert os.path.exists(output_dir_name), 'output directory "{}" does not exist'.format(output_dir_name)

    assert os.path.isdir(output_dir_name), '"{}" is not a valid directory'.format(output_dir_name)

    with ZipFile(reader, "r") as z:
        z.extractall(output_dir_name)


def unzip_all_from_file(zip_file_name: str, output_dir_name: str):
    """Decompress a zip archive file and extract all files to the specified output directory.

    Args:
        zip_file_name: the input zip archive file
        output_dir_name: the output directory for extracted content
    """
    assert os.path.exists(zip_file_name), 'input zip file "{}" does not exist'.format(zip_file_name)
    assert os.path.isfile(zip_file_name), '"{}" is not a valid file'.format(zip_file_name)

    _unzip_all(zip_file_name, output_dir_name)


def unzip_all_from_bytes(data, output_dir_name: str):
    _unzip_all(io.BytesIO(data), output_dir_name)


def convert_legacy_zip(zip_data: bytes) -> bytes:
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
        meta_path = _path_join(folder_name, META_NAME)
        if next((info for info in info_list if info.filename == meta_path), None):
            # Already in job layout
            meta_data = in_zip.read(meta_path)
            meta = json.loads(meta_data)
            if JobMetaKey.JOB_FOLDER_NAME.value not in meta:
                meta[JobMetaKey.JOB_FOLDER_NAME.value] = folder_name
            else:
                return zip_data

        writer = io.BytesIO()
        with ZipFile(writer, "w") as out_zip:
            if meta:
                out_zip.writestr(meta_path, json.dumps(meta))
                out_zip.comment = in_zip.comment  # preserve the comment
                for item in in_zip.infolist():
                    if item.filename != meta_path:
                        out_zip.writestr(item, in_zip.read(item.filename))
            else:
                out_zip.writestr(meta_path, _get_default_meta(folder_name))
                # Push everything else to a sub folder with the same name:
                # hello-pt/README.md -> hello-pt/hello-pt/README.md
                for info in info_list:
                    name = info.filename
                    content = in_zip.read(name)
                    path = folder_name + "/" + name
                    info.filename = path
                    out_zip.writestr(info, content)

        return writer.getvalue()
