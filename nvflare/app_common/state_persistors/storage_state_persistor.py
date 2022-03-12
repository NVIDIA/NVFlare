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

import os
import pickle

from nvflare.apis.fl_snapshot import FLSnapshot
from nvflare.apis.state_persistor import StatePersistor
from nvflare.apis.storage import StorageSpec


class StorageStatePersistor(StatePersistor):
    def __init__(self, storage: StorageSpec, location: str):
        self.storage = storage
        self.location = location

        if not os.path.isabs(location):
            raise ValueError("snapshot location must be absolute path.")

    def save(self, snapshot: FLSnapshot) -> str:
        """Call to save the snapshot of the FL state to storage.
        Args:
            snapshot: FLSnapshot object
        Returns: storage location
        """
        self.storage.create_object(uri=self.location, data=pickle.dumps(snapshot), meta={}, overwrite_existing=True)

        return self.location

    def retrieve(self) -> FLSnapshot:
        """Call to load the persisted FL components snapshot from the persisted location.
        Args:
        Returns: retrieved Snapshot
        """
        retrieved_snapshot = None
        try:
            retrieved_snapshot = pickle.loads(self.storage.get_data(self.location))
        finally:
            return retrieved_snapshot

    def delete(self, location: str):
        """To delete the FL component snapshot.
        Args:
            location: persist location
        Returns:
        """
        self.storage.delete_object(location)
