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
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

from nvflare.apis.job_def import ALL_SITES, JobMetaKey

META_NAME = "meta.json"


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


def normpath_for_zip(path):
    """Normalizes the path for zip file.

    Args:
        path (str): the path to be normalized
    """
    path = os.path.normpath(path)
    path = os.path.splitdrive(path)[1]
    # ZIP spec requires forward slashes
    return path.replace("\\", "/")


def remove_leading_dotdot(path: str) -> str:
    path = str(Path(path))
    while path.startswith(f"..{os.path.sep}"):
        path = path[3:]
    return path


def split_path(path: str) -> (str, str):
    """Splits a path into a pair of head and tail.

    It removes trailing `os.path.sep` and call `os.path.split`

    Args:
        path: Path to split

    Returns:
        A tuple of `(head, tail)`
    """
    path = str(Path(path))
    if path.endswith(os.path.sep):
        full_path = path[:-1]
    else:
        full_path = path

    return os.path.split(full_path)


def get_all_file_paths(directory):
    """Gets all file paths in the directory.

    Args:
        directory: directory to get all paths for

    Returns:
        A list of paths of all the files in the provided directory
    """
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_paths.append(normpath_for_zip(os.path.join(root, filename)))
        for dir_name in directories:
            file_paths.append(normpath_for_zip(os.path.join(root, dir_name)))

    return file_paths


def _zip_directory(root_dir: str, folder_name: str, output_file):
    """Creates a zip archive file for the specified directory.

    Args:
        root_dir: root path that contains the folder to be zipped
        folder_name: path to the folder to be zipped, relative to root_dir
        output_file: file to write to
    """
    dir_name = normpath_for_zip(os.path.join(root_dir, folder_name))
    if not os.path.exists(dir_name):
        raise FileNotFoundError(f'output directory "{dir_name}" does not exist')

    if not os.path.isdir(dir_name):
        raise NotADirectoryError(f'"{dir_name}" is not a valid directory')

    file_paths = get_all_file_paths(dir_name)
    if folder_name:
        prefix_len = len(split_path(dir_name)[0]) + 1
    else:
        prefix_len = len(dir_name) + 1

    # writing files to a zipfile
    with ZipFile(output_file, "w") as z:
        # writing each file one by one
        for full_path in file_paths:
            rel_path = full_path[prefix_len:]
            z.write(full_path, arcname=rel_path)


def zip_directory_to_bytes(root_dir: str, folder_name: str) -> bytes:
    """Compresses a directory and return the bytes value of it.

    Args:
        root_dir: root path that contains the folder to be zipped
        folder_name: path to the folder to be zipped, relative to root_dir
    """
    bio = io.BytesIO()
    _zip_directory(root_dir, folder_name, bio)
    return bio.getvalue()


def unzip_all_from_bytes(zip_data: bytes, output_dir_name: str):
    """Decompresses a zip and extracts all files to the specified output directory.

    Args:
        zip_data: the input zip data
        output_dir_name: the output directory for extracted content
    """
    if not os.path.exists(output_dir_name):
        raise FileNotFoundError(f'output directory "{output_dir_name}" does not exist')

    if not os.path.isdir(output_dir_name):
        raise NotADirectoryError(f'"{output_dir_name}" is not a valid directory')

    with ZipFile(io.BytesIO(zip_data), "r") as z:
        z.extractall(output_dir_name)


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
        meta_path = normpath_for_zip(os.path.join(folder_name, META_NAME))
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
                for info in info_list:
                    if info.filename != meta_path:
                        out_zip.writestr(info, in_zip.read(info.filename))
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
