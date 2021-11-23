# Copyright (c) 2021, NVIDIA CORPORATION.
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
from zipfile import ZipFile


def get_all_file_paths(directory):
    """
    Get all file paths in the directory.
    Args:
        directory: directory to get all paths for

    Returns: all paths in the provided directory
    """
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_paths.append(os.path.join(root, filename))
        for dir_name in directories:
            file_paths.append(os.path.join(root, dir_name))

    return file_paths


def _zip_directory(root_dir: str, folder_name: str, writer):
    """
    Create a zip archive file for the specified directory.

    Args:
        root_dir: root path that contains the folder to be zipped
        folder_name: path to the folder to be zipped, relative to root_dir
    """
    dir_name = os.path.join(root_dir, folder_name)
    assert os.path.exists(dir_name), 'directory "{}" does not exist'.format(dir_name)
    assert os.path.isdir(dir_name), '"{}" is not a valid directory'.format(dir_name)

    file_paths = get_all_file_paths(dir_name)

    # writing files to a zipfile
    with ZipFile(writer, "w") as z:
        # writing each file one by one
        for full_path in file_paths:
            rel_path = os.path.relpath(full_path, root_dir)
            z.write(full_path, arcname=rel_path)


def zip_directory_to_file(root_dir: str, folder_name: str, output_file_name: str):
    """
    Create a zip archive file for the specified directory.

    Args:
        root_dir: root path that contains the folder to be zipped
        folder_name: path to the folder to be zipped, relative to root_dir
        output_file_name: name of the output file
    """
    _zip_directory(root_dir, folder_name, output_file_name)


def zip_directory_to_bytes(root_dir: str, folder_name: str):
    bio = io.BytesIO()
    _zip_directory(root_dir, folder_name, bio)
    return bio.getvalue()


def _unzip_all(reader, output_dir_name: str):
    """
    Decompress a zip archive file and extract all files to the specified output directory.

    Args:
        reader: the input zip reader
        output_dir_name: the output directory for extracted content
    """
    assert os.path.exists(output_dir_name), 'output directory "{}" does not exist'.format(output_dir_name)

    assert os.path.isdir(output_dir_name), '"{}" is not a valid directory'.format(output_dir_name)

    with ZipFile(reader, "r") as z:
        z.extractall(output_dir_name)


def unzip_all_from_file(zip_file_name: str, output_dir_name: str):
    """
    Decompress a zip archive file and extract all files to the specified output directory.

    Args:
        zip_file_name: the input zip archive file
        output_dir_name: the output directory for extracted content
    """
    assert os.path.exists(zip_file_name), 'input zip file "{}" does not exist'.format(zip_file_name)
    assert os.path.isfile(zip_file_name), '"{}" is not a valid file'.format(zip_file_name)

    _unzip_all(zip_file_name, output_dir_name)


def unzip_all_from_bytes(data, output_dir_name: str):
    _unzip_all(io.BytesIO(data), output_dir_name)
