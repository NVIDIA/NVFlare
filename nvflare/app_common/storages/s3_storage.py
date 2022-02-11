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
import inspect
import io
import os
from functools import wraps
from typing import ByteString, List, Tuple

from minio import Minio
from minio.commonconfig import REPLACE, CopySource

from nvflare.apis.storage import StorageSpec


def validate_class_methods_args(cls):
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        if name != "__init_subclass__":
            setattr(cls, name, validate_args(method))
    return cls


def validate_args(method):
    signature = inspect.signature(method)

    @wraps(method)
    def wrapper(*args, **kwargs):
        bound_arguments = signature.bind(*args, **kwargs)
        for name, value in bound_arguments.arguments.items():
            annotation = signature.parameters[name].annotation
            if not (annotation is inspect.Signature.empty or isinstance(value, annotation)):
                raise TypeError(
                    "argument '{}' of {} must be {} but got {}".format(name, method, annotation, type(value))
                )
        return method(*args, **kwargs)

    return wrapper


@validate_class_methods_args
class S3Storage(StorageSpec):
    def __init__(self, endpoint, access_key, secret_key, secure, bucket_name):
        self.s3_client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = bucket_name

    def _object_exists(self, uri: str):
        try:
            self.s3_client.stat_object(self.bucket_name, uri)
        except Exception as e:
            if e.message == "Object does not exist":
                return False
            raise Exception(e.message)
        return True

    def create_object(self, uri: str, data: ByteString, meta: dict, overwrite_existing: bool = False):
        """Create a new object or update an existing object

        Args:
            uri: URI of the object
            data: content of the object
            meta: meta info of the object
            overwrite_existing: whether to overwrite the object if already exists

        Returns:

        Raises exception when:

        - invalid URI specification
        - invalid args
        - object already exists and overwrite_existing is False
        - error creating the object

        Examples of URI:

        /state/engine/...
        /runs/approved/covid_exam.3
        /runs/pending/splee_seg.1

        """
        if self._object_exists(uri) and not overwrite_existing:
            raise Exception("object {} already exists and overwrite_existing is False".format(uri))

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

        Raises exception when:

        - no such object
        - invalid args
        - error updating the object

        """
        if not self._object_exists(uri):
            raise Exception("object {} does not exist".format(uri))

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

        Raises exception when:

        - no such object
        - invalid args
        - error updating the object

        """
        if not self._object_exists(uri):
            raise Exception("object {} does not exist".format(uri))

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

        """
        dir_path = dir_path.rstrip("/") + "/"
        return [
            "/" + obj._object_name
            for obj in list(self.s3_client.list_objects(self.bucket_name, prefix=dir_path, include_user_meta=True))
            if obj._metadata
        ]

    def get_meta(self, uri: str) -> dict:
        """Get user defined meta info of the specified object

        Args:
            uri: URI of the object

        Returns: meta info of the object.

        Raises exception when:

        - no such object

        """
        if not self._object_exists(uri):
            raise Exception("object {} does not exist".format(uri))

        return ast.literal_eval(self.s3_client.stat_object(self.bucket_name, uri)._metadata["x-amz-meta-user_metadata"])

    def get_full_meta(self, uri: str) -> dict:
        """Get full meta info of the specified object

        Args:
            uri: URI of the object

        Returns: meta info of the object.

        Raises exception when:

        - no such object

        """
        if not self._object_exists(uri):
            raise Exception("object {} does not exist".format(uri))

        return self.s3_client.stat_object(self.bucket_name, uri)

    def get_data(self, uri: str) -> bytes:
        """Get data of the specified object

        Args:
            uri: URI of the object

        Returns: data of the object.

        Raises exception when:

        - no such object

        """
        if not self._object_exists(uri):
            raise Exception("object {} does not exist".format(uri))

        return self.s3_client.get_object(self.bucket_name, uri).data

    def get_detail(self, uri: str) -> Tuple[dict, bytes]:
        """Get both data and meta of the specified object

        Args:
            uri: URI of the object

        Returns: meta info and data of the object.

        Raises exception when:

        - no such object

        """
        if not os.path.exists(uri):
            raise Exception("object {} does not exist".format(uri))

        return self.get_meta(uri), self.get_data(uri)

    def delete_object(self, uri: str):
        """Delete specified object

        Args:
            uri: URI of the object

        Returns:

        Raises exception when:

        - no such object

        """
        if not self._object_exists(uri):
            raise Exception("object {} does not exist".format(uri))

        self.s3_client.remove_object(self.bucket_name, uri)
