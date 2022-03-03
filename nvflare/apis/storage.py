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

from typing import ByteString, List, Tuple


class StorageSpec(object):

    """
    This defines the functional spec of object storage.
    Object - an object is identified by a URI (unique resource identifier).
    An object has content (data)
    An object has meta info that describes the control info of the object.
    """

    def create_object(self, uri: str, data: ByteString, meta: dict, overwrite_existing: bool):
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
        pass

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
        pass

    def update_data(self, uri: str, data: ByteString):
        """Update the meta info of the specified object

        Args:
            uri: URI of the object
            data: value of new data

        Returns:

        Raises exception when:

        - no such object
        - invalid args
        - error updating the object

        """
        pass

    def list_objects(self, path: str) -> List[str]:
        """List all objects in the specified path.

        Args:
            path: the path to the objects

        Returns: list of URIs of objects

        """
        pass

    def get_meta(self, uri: str) -> dict:
        """Get user defined meta info of the specified object

        Args:
            uri: URI of the object

        Returns: meta info of the object.

        Raises exception when:

        - no such object

        """
        pass

    def get_full_meta(self, uri: str) -> dict:
        """Get full meta info of the specified object

        Args:
            uri: URI of the object

        Returns: meta info of the object.

        Raises exception when:

        - no such object

        """
        pass

    def get_data(self, uri: str) -> bytes:
        """Get data of the specified object

        Args:
            uri: URI of the object

        Returns: data of the object.

        Raises exception when:

        - no such object

        """
        pass

    def get_detail(self, uri: str) -> Tuple[dict, bytes]:
        """Get both data and meta of the specified object

        Args:
            uri: URI of the object

        Returns: meta info and data of the object.

        Raises exception when:

        - no such object

        """
        pass

    def delete_object(self, uri: str):
        """Delete specified object

        Args:
            uri: URI of the object

        Returns:

        Raises exception when:

        - no such object

        """
        pass
