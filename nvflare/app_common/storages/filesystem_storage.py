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

import ast
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple

from nvflare.apis.storage import DATA, META, StorageException, StorageSpec
from nvflare.apis.utils.format_check import validate_class_methods_args
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)


def _write(path: str, content, mv_file=True):
    """Create a file at the specified 'path' with the specified 'content'.

    Args:
        path: the path of the file to be created
        content: content for the file to be created. It could be either bytes, or path (str) to the source file that
            contains the content.
        mv_file: whether the destination file should be created simply by moving the source file. This is applicable
            only when the 'content' is the path of the source file. If mv_file is False, the destination is created
            by copying from the source file, and the source file will remain intact; If mv_file is True, the
            destination file is created by "move" the source file, and the original source file will no longer exist.

    Returns:

    """
    tmp_path = path + "_" + str(uuid.uuid4())
    try:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            with open(tmp_path, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
        elif isinstance(content, str):
            # this is the name of the file that contains content
            if not os.path.exists(content):
                raise FileNotFoundError(f"file {content} does not exist")
            if not os.path.isfile(content):
                raise ValueError(f"{content} is not a valid file")
            if mv_file:
                shutil.move(content, tmp_path)
            else:
                shutil.copyfile(content, tmp_path)
        else:
            raise RuntimeError(f"content must be bytes or str but got {type(content)}")
    except Exception as e:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise StorageException(f"failed to write content: {secure_format_exception(e)}")

    if os.path.exists(tmp_path):
        os.rename(tmp_path, path)


def _read(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            content = f.read()
    except Exception as e:
        raise StorageException(f"failed to read content: {secure_format_exception(e)}")

    return content


def _object_exists(uri: str):
    """Checks whether an object exists at specified directory."""
    data_exists = os.path.isfile(os.path.join(uri, "data"))
    meta_exists = os.path.isfile(os.path.join(uri, "meta"))
    return all((os.path.isabs(uri), os.path.isdir(uri), data_exists, meta_exists))


def _encode_meta(meta: dict) -> bytes:
    return json.dumps(meta).encode("utf-8")


def _decode_meta(data: bytes) -> dict:
    s = data.decode("utf-8")
    if s.startswith('"'):
        # this is in old format
        result = ast.literal_eval(json.loads(s))
    else:
        # this is json string
        result = json.loads(s)
    return result


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

    def _object_path(self, uri: str):
        return os.path.join(self.root_dir, uri.lstrip(self.uri_root))

    def create_object(self, uri: str, data, meta: dict, overwrite_existing: bool = False):
        """Creates an object.

        Args:
            uri: URI of the object
            data: content of the object; bytes or file name that contains data
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
        full_uri = self._object_path(uri)

        if _object_exists(full_uri) and not overwrite_existing:
            raise StorageException(f"object {uri} already exists and overwrite_existing is False")

        if not _object_exists(full_uri) and os.path.isdir(full_uri) and os.listdir(full_uri):
            raise StorageException(f"cannot create object {uri} at nonempty directory")

        data_path = os.path.join(full_uri, DATA)
        meta_path = os.path.join(full_uri, META)
        _write(data_path, data)
        try:
            _write(meta_path, _encode_meta(meta))
        except Exception as e:
            os.remove(data_path)
            raise e
        return full_uri

    def clone_object(self, from_uri: str, to_uri: str, meta: dict, overwrite_existing: bool = False):
        full_uri = self._object_path(to_uri)

        if _object_exists(full_uri) and not overwrite_existing:
            raise StorageException(f"object {to_uri} already exists and overwrite_existing is False")

        if not _object_exists(full_uri) and os.path.isdir(full_uri) and os.listdir(full_uri):
            raise StorageException(f"cannot create object {to_uri} at nonempty directory")

        data_path = os.path.join(full_uri, DATA)

        from_full_uri = self._object_path(from_uri)
        from_data_path = os.path.join(from_full_uri, DATA)
        _write(data_path, from_data_path, mv_file=False)

        meta_path = os.path.join(full_uri, META)
        try:
            _write(meta_path, _encode_meta(meta))
        except Exception as e:
            os.remove(data_path)
            raise e
        return full_uri

    def update_object(self, uri: str, data, component_name: str = DATA):
        """Update the object

        Args:
            uri: URI of the object
            data: content data of the component
            component_name: component name

        Raises StorageException when the object does not exit.

        """
        full_dir_path = self._object_path(uri)
        if not os.path.isdir(full_dir_path):
            raise StorageException(f"path {full_dir_path} is not a valid directory.")

        if not StorageSpec.is_valid_component(component_name):
            raise StorageException(f"{component_name } is not a valid component for storage object.")

        component_path = os.path.join(full_dir_path, component_name)
        _write(component_path, data)

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
        full_uri = self._object_path(uri)

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        if replace:
            _write(os.path.join(full_uri, META), _encode_meta(meta))
        else:
            prev_meta = self.get_meta(uri)
            prev_meta.update(meta)
            _write(os.path.join(full_uri, META), _encode_meta(prev_meta))

    def list_objects(self, path: str, without_tag=None) -> List[str]:
        """List all objects in the specified path.

        Args:
            path: the path uri to the objects
            without_tag: if set, skip the objects with this specified tag

        Returns:
            list of URIs of objects

        Raises:
            TypeError: if invalid argument types
            StorageException: if path does not exist or is not a valid directory.

        """
        full_dir_path = self._object_path(path)
        if not os.path.isdir(full_dir_path):
            raise StorageException(f"path {full_dir_path} is not a valid directory.")

        result = []

        # Use scandir instead of listdir.
        # According to https://peps.python.org/pep-0471/#os-scandir, scandir is more memory-efficient than listdir
        # when iterating very large directories.
        gen = os.scandir(full_dir_path)
        for e in gen:
            # assert isinstance(e, os.DirEntry)
            obj_dir = os.path.join(full_dir_path, e.name)
            if _object_exists(obj_dir):
                if not without_tag or not os.path.exists(os.path.join(obj_dir, without_tag)):
                    result.append(os.path.join(path, e.name))
        return result

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
        full_uri = self._object_path(uri)

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        return _decode_meta(_read(os.path.join(full_uri, META)))

    def get_data(self, uri: str, component_name: str = DATA) -> bytes:
        """Gets data of the specified object.

        Args:
            uri: URI of the object
            component_name: storage component name

        Returns:
            data of the object.

        Raises:
            TypeError: if invalid argument types
            StorageException: if object does not exist

        """
        full_uri = self._object_path(uri)

        if not StorageSpec.is_valid_component(component_name):
            raise StorageException(f"{component_name } is not a valid component for storage object.")

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        return _read(os.path.join(full_uri, component_name))

    def get_data_for_download(self, uri: str, component_name: str = DATA, download_file: str = None):
        full_uri = self._object_path(uri)

        if not StorageSpec.is_valid_component(component_name):
            raise StorageException(f"{component_name } is not a valid component for storage object.")

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        if os.path.exists(download_file):
            os.remove(download_file)
        src = os.path.join(full_uri, component_name)
        if os.path.exists(src):
            os.symlink(src, download_file)
        else:
            log.info(f"{src} does not exist, skipping the creation of the symlink {download_file} for download.")

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
        full_uri = self._object_path(uri)

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
        full_uri = self._object_path(uri)

        if not _object_exists(full_uri):
            raise StorageException("object {} does not exist".format(uri))

        shutil.rmtree(full_uri)

        return full_uri

    def tag_object(self, uri: str, tag: str, data=None):
        full_path = self._object_path(uri)
        mark_file = os.path.join(full_path, tag)
        with open(mark_file, "w") as f:
            if data:
                f.write(data)
