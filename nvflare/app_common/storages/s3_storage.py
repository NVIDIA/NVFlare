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
import io
import os
from pathlib import Path
from typing import ByteString, List, Tuple

from minio import Minio
from minio.commonconfig import REPLACE, CopySource

from nvflare.apis.storage import StorageSpec
from nvflare.apis.utils.format_check import validate_class_methods_args

URI_ROOT = os.path.abspath(os.sep)


@validate_class_methods_args
class S3Storage(StorageSpec):
    def __init__(self, endpoint, access_key, secret_key, secure, bucket_name):
        """Init S3Storage.

        Uses S3 bucket to persist objects, with absolute paths as object URIs.

        Args:
            endpoint: hostname of S3 service
            access_key: access key (username) of S3 service account
            secret_key: secret key (password) of S3 service account
            secure: flag for secure (TLS) mode
            bucket_name: name of S3 bucket

        """

        self.s3_client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        if not self.s3_client.bucket_exists(bucket_name):
            self.s3_client.make_bucket(bucket_name)
        self.bucket_name = bucket_name

    def _object_exists(self, uri: str):
        try:
            self.s3_client.stat_object(self.bucket_name, uri)
        except:
            return False
        return True

    def create_object(self, uri: str, data: ByteString, meta: dict, overwrite_existing: bool = False):
        """Create a new object or update an existing object

        Args:
            uri: URI of the object
            data: content of the object
            meta: meta info of the object
            overwrite_existing: whether to overwrite the object if already exists

        Returns:

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if error creating the object
                - if object already exists and overwrite_existing is False
                - if object will be inside prexisiting object
                - if object will be at a non-empty directory

        Examples of URI:

        /state/engine/...
        /runs/approved/covid_exam.3
        /runs/pending/splee_seg.1

        """
        if self._object_exists(uri) and not overwrite_existing:
            raise RuntimeError("object {} already exists and overwrite_existing is False".format(uri))

        path_parts = Path(uri).parts
        for i in range(2, len(path_parts)):
            parent_path = str(Path(*path_parts[0:i]))
            if self._object_exists(parent_path):
                raise RuntimeError("cannot create object {} inside preexisting object {}".format(uri, parent_path))

        if not self._object_exists(uri) and (
            list(self.s3_client.list_objects(self.bucket_name, prefix=uri, include_user_meta=True))
        ):
            raise RuntimeError("cannot create object {} at nonempty directory".format(uri))

        self.s3_client.put_object(
            self.bucket_name, uri, data=io.BytesIO(data), length=-1, metadata={"user_metadata": meta}, part_size=5242880
        )

    def update_meta(self, uri: str, meta: dict, replace: bool):
        """Update the meta info of the specified object

        Args:
            uri: URI of the object
            meta: value of new meta info
            replace: whether to replace the current meta completely or partial update

        Returns:

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise RuntimeError("object {} does not exist".format(uri))

        if not replace:
            prev_meta = self.get_meta(uri)
            prev_meta.update(meta)
            meta = prev_meta

        self.s3_client.copy_object(
            self.bucket_name,
            uri,
            CopySource(self.bucket_name, uri),
            metadata={"user_metadata": meta},
            metadata_directive=REPLACE,
        )

    def update_data(self, uri: str, data: ByteString):
        """Update the data info of the specified object

        Args:
            uri: URI of the object
            data: value of new data

        Returns:

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise RuntimeError("object {} does not exist".format(uri))

        self.s3_client.put_object(
            self.bucket_name,
            uri,
            data=io.BytesIO(data),
            length=-1,
            metadata={"user_metadata": self.get_meta(uri)},
            part_size=5242880,
        )

    def list_objects(self, dir_path: str) -> List[str]:
        """List all objects in the specified path.

        Args:
            path: the path to the objects

        Returns: list of URIs of objects

        Raises:
            TypeError: if invalid argument types

        """
        dir_path = dir_path.rstrip(os.sep) + os.sep
        return [
            URI_ROOT + obj._object_name
            for obj in list(self.s3_client.list_objects(self.bucket_name, prefix=dir_path, include_user_meta=True))
            if obj._metadata
        ]

    def get_meta(self, uri: str) -> dict:
        """Get user defined meta info of the specified object

        Args:
            uri: URI of the object

        Returns: meta info of the object.

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise RuntimeError("object {} does not exist".format(uri))

        return ast.literal_eval(self.s3_client.stat_object(self.bucket_name, uri)._metadata["x-amz-meta-user_metadata"])

    def get_full_meta(self, uri: str) -> dict:
        """Get full meta info of the specified object

        Args:
            uri: URI of the object

        Returns: meta info of the object.

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise RuntimeError("object {} does not exist".format(uri))

        return self.s3_client.stat_object(self.bucket_name, uri)

    def get_data(self, uri: str) -> bytes:
        """Get data of the specified object

        Args:
            uri: URI of the object

        Returns: data of the object.

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise RuntimeError("object {} does not exist".format(uri))

        return self.s3_client.get_object(self.bucket_name, uri).data

    def get_detail(self, uri: str) -> Tuple[dict, bytes]:
        """Get both data and meta of the specified object

        Args:
            uri: URI of the object

        Returns: meta info and data of the object.

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise RuntimeError("object {} does not exist".format(uri))

        return self.get_meta(uri), self.get_data(uri)

    def delete_object(self, uri: str):
        """Delete specified object

        Args:
            uri: URI of the object

        Returns:

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise RuntimeError("object {} does not exist".format(uri))

        self.s3_client.remove_object(self.bucket_name, uri)
