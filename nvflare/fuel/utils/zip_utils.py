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
import os
from pathlib import Path
from zipfile import ZipFile


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
        raise FileNotFoundError(f'source directory "{dir_name}" does not exist')

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


def ls_zip_from_bytes(zip_data: bytes):
    """Returns info of a zip.

    Args:
        zip_data: the input zip data
    """
    with ZipFile(io.BytesIO(zip_data), "r") as z:
        return z.infolist()


def unzip_single_file_from_bytes(zip_data: bytes, output_dir_name: str, file_path: str):
    """Decompresses a zip and extracts single specified file to the specified output directory.

    Args:
        zip_data: the input zip data
        output_dir_name: the output directory for extracted content
        file_path: file path to file to unzip
    """
    path_to_file, _ = split_path(file_path)
    output_dir_name = os.path.join(output_dir_name, path_to_file)
    os.makedirs(output_dir_name)
    if not os.path.exists(output_dir_name):
        raise FileNotFoundError(f'output directory "{output_dir_name}" does not exist')

    if not os.path.isdir(output_dir_name):
        raise NotADirectoryError(f'"{output_dir_name}" is not a valid directory')

    with ZipFile(io.BytesIO(zip_data), "r") as z:
        z.extract(file_path, path=output_dir_name)


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
