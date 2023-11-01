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

from abc import ABC, abstractmethod
from typing import List, Tuple

DATA = "data"
JOB_ZIP = "job.zip"
META = "meta"
META_JSON = "meta.json"
WORKSPACE = "workspace"
WORKSPACE_ZIP = "workspace.zip"
MANIFEST = "manifest.json"


class StorageException(Exception):
    """Base class for Storage exceptions."""

    pass


class StorageSpec(ABC):
    """Functional spec of object storage.

    An object is identified by a URI (unique resource identifier).
    Each object contains:

        - content (data)
        - meta info that describes the control info of the object.

    """

    @abstractmethod
    def create_object(self, uri: str, data, meta: dict, overwrite_existing: bool):
        """Creates an object.

        Examples of URI:

            /state/engine/...
            /runs/approved/covid_exam.3
            /runs/pending/spleen_seg.1

        Args:
            uri: URI of the object
            data: content of the object
            meta: meta info of the object
            overwrite_existing: whether to overwrite the object if already exists

        Raises StorageException when:
            - invalid args
            - object already exists and overwrite_existing is False
            - error creating the object

        """
        pass

    @abstractmethod
    def update_object(self, uri: str, data, component_name: str):
        """Update the object

        Args:
            uri: URI of the object
            data: content data of the component, or the content file location
            component_name: component name

        Raises StorageException when the object does not exit.

        """
        pass

    @abstractmethod
    def update_meta(self, uri: str, meta: dict, replace: bool):
        """Updates the meta info of the specified object.

        Args:
            uri: URI of the object
            meta: value of new meta info
            replace: whether to replace the current meta completely or partial update

        Raises StorageException when:
            - invalid args
            - no such object
            - error updating the object

        """
        pass

    @abstractmethod
    def list_objects(self, path: str) -> List[str]:
        """Lists all objects in the specified path.

        Args:
            path: the path to the objects

        Returns:
            list of URIs of objects

        """
        pass

    @abstractmethod
    def get_meta(self, uri: str) -> dict:
        """Gets user defined meta info of the specified object.

        Args:
            uri: URI of the object

        Returns:
            meta info of the object.
            if object does not exist, return empty dict {}

        Raises StorageException when:
          - invalid args

        """
        pass

    @abstractmethod
    def get_data(self, uri: str, component_name: str = DATA) -> bytes:
        """Gets data of the specified object.

        Args:
            uri: URI of the object
            component_name: storage component name

        Returns:
            data of the object.
            if object does not exist, return None

        Raises StorageException when:
            - invalid args

        """
        pass

    @abstractmethod
    def get_data_for_download(self, uri: str, component_name: str = DATA, download_file: str = None):
        """Gets data of the specified object.

        Args:
            uri: URI of the object
            component_name: storage component name
            download_file: component file_name for download

        Raises StorageException when:
            - invalid args

        """
        pass

    @abstractmethod
    def get_detail(self, uri: str) -> Tuple[dict, bytes]:
        """Gets both data and meta of the specified object.

        Args:
            uri: URI of the object

        Returns:
            meta info and data of the object.

        Raises StorageException when:
            - invalid args
            - no such object

        """
        pass

    @abstractmethod
    def delete_object(self, uri: str):
        """Deletes specified object.

        Args:
            uri: URI of the object

        """
        pass

    @staticmethod
    def is_valid_component(component_name):
        return component_name in [DATA, META, WORKSPACE, MANIFEST]
