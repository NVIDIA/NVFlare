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
from typing import List, Tuple

from minio import Minio
from minio.commonconfig import REPLACE, CopySource
from minio.error import MinioException

from nvflare.apis.storage import StorageException, StorageSpec
from nvflare.apis.utils.format_check import validate_class_methods_args

URI_ROOT = os.path.abspath(os.sep)
_USER_META_KEY = "user_metadata"
_AWS_PART_SIZE = 5 * 1024 * 1024


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

        Raises:
            RuntimeError: error creating minio S3 client and bucket

        """
        try:
            self.s3_client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
            if not self.s3_client.bucket_exists(bucket_name):
                self.s3_client.make_bucket(bucket_name)
        except MinioException as e:
            raise StorageException("Error creating minio s3 client: {}".format(str(e)))
        self.bucket_name = bucket_name

    def _object_exists(self, uri: str):
        try:
            self.s3_client.stat_object(self.bucket_name, uri)
        except MinioException:
            return False
        return True

    def create_object(self, uri: str, data: bytes, meta: dict, overwrite_existing: bool = False):
        """Creates an object.

        Args:
            uri: URI of the object
            data: content of the object
            meta: meta info of the object
            overwrite_existing: whether to overwrite the object if already exists

        Raises:
            TypeError: if invalid argument types
            RuntimeError:
                - if error creating the object
                - if object already exists and overwrite_existing is False
                - if object will be inside pre-existing object
                - if object will be at a non-empty directory

        Examples of URI:

        /state/engine/...
        /runs/approved/covid_exam.3
        /runs/pending/spleen_seg.1

        """
        if self._object_exists(uri) and not overwrite_existing:
            raise StorageException("object {} already exists and overwrite_existing is False".format(uri))

        if not self._object_exists(uri):
            try:
                dir_is_nonempty = list(
                    self.s3_client.list_objects(self.bucket_name, prefix=uri, include_user_meta=True)
                )
            except MinioException as e:
                raise StorageException(
                    "cannot create object: checking if uri has any objects failed: {}".format(str(e))
                )
            if dir_is_nonempty:
                raise StorageException("cannot create object {} at nonempty directory".format(uri))

        try:
            self.s3_client.put_object(
                self.bucket_name,
                uri,
                data=io.BytesIO(data),
                length=-1,
                metadata={_USER_META_KEY: meta},
                part_size=_AWS_PART_SIZE,
            )
        except MinioException as e:
            raise StorageException("error putting object into bucket: {}".format(str(e)))

    def update_meta(self, uri: str, meta: dict, replace: bool):
        """Updates the meta info of the specified object.

        Args:
            uri: URI of the object
            meta: value of new meta info
            replace: whether to replace the current meta completely or partial update

        Raises:
            TypeError: if invalid argument types
            RuntimeError:
                - if object does not exist
                - if error copying object

        """
        if not self._object_exists(uri):
            raise StorageException("object {} does not exist".format(uri))

        if not replace:
            prev_meta = self.get_meta(uri)
            prev_meta.update(meta)
            meta = prev_meta

        try:
            self.s3_client.copy_object(
                self.bucket_name,
                uri,
                CopySource(self.bucket_name, uri),
                metadata={_USER_META_KEY: meta},
                metadata_directive=REPLACE,
            )
        except MinioException as e:
            raise StorageException("error copying object: {}".format(str(e)))

    def update_data(self, uri: str, data: bytes):
        """Updates the data info of the specified object.

        Args:
            uri: URI of the object
            data: value of new data

        Raises:
            TypeError: if invalid argument types
            RuntimeError:
                - if object does not exist
                - if error putting object into bucket

        """
        if not self._object_exists(uri):
            raise StorageException("object {} does not exist".format(uri))

        try:
            self.s3_client.put_object(
                self.bucket_name,
                uri,
                data=io.BytesIO(data),
                length=-1,
                metadata={_USER_META_KEY: self.get_meta(uri)},
                part_size=_AWS_PART_SIZE,
            )
        except MinioException as e:
            raise StorageException("error putting object into bucket: {}".format(str(e)))

    def list_objects(self, path: str) -> List[str]:
        """List all objects in the specified path.

        Args:
            path: the path to the objects

        Returns:
            list of URIs of objects

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if error listing objects

        """
        dir_path = path.rstrip(os.sep) + os.sep
        try:
            return [
                URI_ROOT + obj._object_name
                for obj in list(self.s3_client.list_objects(self.bucket_name, prefix=dir_path, include_user_meta=True))
                if obj._metadata
            ]
        except MinioException as e:
            raise StorageException(f"error listing objects at {path}: {str(e)}.")

    def get_meta(self, uri: str) -> dict:
        """Gets user defined meta info of the specified object.

        Args:
            uri: URI of the object

        Returns:
            meta info of the object.
            if object does not exist, return empty dict {}

        Raises:
            TypeError: if invalid argument types
            RuntimeError:
                - if error accessing object

        """
        if not self._object_exists(uri):
            raise StorageException("object {} does not exist".format(uri))

        try:
            return ast.literal_eval(
                self.s3_client.stat_object(self.bucket_name, uri)._metadata["x-amz-meta-{}".format(_USER_META_KEY)]
            )
        except MinioException as e:
            raise StorageException(f"error accessing object ({uri}): {str(e)}.")

    def get_data(self, uri: str) -> bytes:
        """Gets data of the specified object.

        Args:
            uri: URI of the object

        Returns:
            data of the object.
            if object does not exist, return None

        Raises:
            TypeError: if invalid argument types
            RuntimeError:
                - if error accessing object

        """
        if not self._object_exists(uri):
            raise StorageException("object {} does not exist".format(uri))

        try:
            return self.s3_client.get_object(self.bucket_name, uri).data
        except MinioException as e:
            raise StorageException(f"error accessing object ({uri}): {str(e)}.")

    def get_detail(self, uri: str) -> Tuple[dict, bytes]:
        """Gets both data and meta of the specified object.

        Args:
            uri: URI of the object

        Returns:
            meta info and data of the object.

        Raises:
            TypeError: if invalid argument types
            RuntimeError: if object does not exist

        """
        if not self._object_exists(uri):
            raise StorageException("object {} does not exist".format(uri))

        return self.get_meta(uri), self.get_data(uri)

    def delete_object(self, uri: str):
        """Deletes the specified object.

        Args:
            uri: URI of the object

        Raises:
            TypeError: if invalid argument types
            RuntimeError:
                - if object does not exist
                - if error removing object

        """
        if not self._object_exists(uri):
            raise StorageException("object {} does not exist".format(uri))

        try:
            self.s3_client.remove_object(self.bucket_name, uri)
        except MinioException as e:
            raise StorageException(f"error removing object ({uri}): {str(e)}.")
