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

import ast
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple

from nvflare.apis.storage import StorageException, StorageSpec
from nvflare.apis.utils.format_check import validate_class_methods_args


def _write(path: str, content):
    tmp_path = path + "_" + str(uuid.uuid4())
    try:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise StorageException("failed to write content: {}".format(e))

    if os.path.exists(tmp_path):
        os.rename(tmp_path, path)


def _read(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            content = f.read()
    except Exception as e:
        raise StorageException("failed to read content: {}".format(e))

    return content


def _object_exists(uri: str):
    """Checks whether an object exists at specified directory."""
    data_exists = os.path.isfile(os.path.join(uri, "data"))
    meta_exists = os.path.isfile(os.path.join(uri, "meta"))
    return all((os.path.isabs(uri), os.path.isdir(uri), data_exists, meta_exists))


@validate_class_methods_args
class FilesystemStorage(StorageSpec):
    def __init__(self, root_dir=os.path.abspath(os.sep), uri_root="/"):
        """Init FileSystemStorage.

        Uses local filesystem to persist objects, with absolute paths as object URIs.

        Args:
            root_dir: the absolute path on the filesystem to store things
            uri_root: serving as the root of the storage. All URIs are rooted at this uri_root.
        """
        if not os.path.isabs(root_dir):
            raise ValueError(f"root_dir {root_dir} must be an absolute path.")
        if os.path.exists(root_dir) and not os.path.isdir(root_dir):
            raise ValueError(f"root_dir {root_dir} exists but is not a directory.")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=False)
        self.root_dir = root_dir
        self.uri_root = uri_root

    def create_object(self, uri: str, data: bytes, meta: dict, overwrite_existing: bool = False):
        """Creates an object.

        Args:
            uri: URI of the object
            data: content of the object
            meta: meta of the object
            overwrite_existing: whether to overwrite the object if already exists

        Raises:
            TypeError: if invalid argument types
            StorageException:
                - if error creating the object
                - if object already exists and overwrite_existing is False
                - if object will be at a non-empty directory
            IOError: if error writing the object

        """
        full_uri = os.path.join(self.root_dir, uri.lstrip(self.uri_root))

        if _object_exists(full_uri) and not overwrite_existing:
            raise StorageException("object {} already exists and overwrite_existing is False".format(uri))

        if not _object_exists(full_uri) and os.path.isdir(full_uri) and os.listdir(full_uri):
            raise StorageException("cannot create object {} at nonempty directory".format(uri))

        data_path = os.path.join(full_uri, "data")
        meta_path = os.path.join(full_uri, "meta")

        tmp_data_path = data_path + "_" + str(uuid.uuid4())
        _write(tmp_data_path, data)
        try:
            _write(meta_path, json.dumps(str(meta)).encode("utf-8"))
        except Exception as e:
            os.remove(tmp_data_path)
            raise e
        os.rename(tmp_data_path, data_path)

        return full_uri

    def update_meta(self, uri: str, meta: dict, replace: bool):
        """Updates the meta of the specified object.

        Args:
            uri: URI of the object
            meta: value of new meta
            replace: whether to replace the current meta completely or partial update

        Raises:
            TypeError: if invalid argument types
            StorageException: if object does not exist
            IOError: if error writing the object

        """
        full_uri = os.path.join(self.root_dir, uri.lstrip(self.uri_root))

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        if replace:
            _write(os.path.join(full_uri, "meta"), json.dumps(str(meta)).encode("utf-8"))
        else:
            prev_meta = self.get_meta(uri)
            prev_meta.update(meta)
            _write(os.path.join(full_uri, "meta"), json.dumps(str(prev_meta)).encode("utf-8"))

    def update_data(self, uri: str, data: bytes):
        """Updates the data of the specified object.

        Args:
            uri: URI of the object
            data: value of new data

        Raises:
            TypeError: if invalid argument types
            StorageException: if object does not exist
            IOError: if error writing the object

        """
        full_uri = os.path.join(self.root_dir, uri.lstrip(self.uri_root))

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        _write(os.path.join(full_uri, "data"), data)

    def list_objects(self, path: str) -> List[str]:
        """List all objects in the specified path.

        Args:
            path: the path uri to the objects

        Returns:
            list of URIs of objects

        Raises:
            TypeError: if invalid argument types
            StorageException: if path does not exist or is not a valid directory.

        """
        full_dir_path = os.path.join(self.root_dir, path.lstrip(self.uri_root))
        if not os.path.isdir(full_dir_path):
            raise StorageException(f"path {full_dir_path} is not a valid directory.")

        return [
            os.path.join(path, f) for f in os.listdir(full_dir_path) if _object_exists(os.path.join(full_dir_path, f))
        ]

    def get_meta(self, uri: str) -> dict:
        """Gets meta of the specified object.

        Args:
            uri: URI of the object

        Returns:
            meta of the object.

        Raises:
            TypeError: if invalid argument types
            StorageException: if object does not exist

        """
        full_uri = os.path.join(self.root_dir, uri.lstrip(self.uri_root))

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        return ast.literal_eval(json.loads(_read(os.path.join(full_uri, "meta")).decode("utf-8")))

    def get_data(self, uri: str) -> bytes:
        """Gets data of the specified object.

        Args:
            uri: URI of the object

        Returns:
            data of the object.

        Raises:
            TypeError: if invalid argument types
            StorageException: if object does not exist

        """
        full_uri = os.path.join(self.root_dir, uri.lstrip(self.uri_root))

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        return _read(os.path.join(full_uri, "data"))

    def get_detail(self, uri: str) -> Tuple[dict, bytes]:
        """Gets both data and meta of the specified object.

        Args:
            uri: URI of the object

        Returns:
            meta and data of the object.

        Raises:
            TypeError: if invalid argument types
            StorageException: if object does not exist

        """
        full_uri = os.path.join(self.root_dir, uri.lstrip(self.uri_root))

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        return self.get_meta(uri), self.get_data(uri)

    def delete_object(self, uri: str):
        """Deletes the specified object.

        Args:
            uri: URI of the object

        Raises:
            TypeError: if invalid argument types
            StorageException: if object does not exist

        """
        full_uri = os.path.join(self.root_dir, uri.lstrip(self.uri_root))

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        shutil.rmtree(full_uri)

        return full_uri
