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

from nvflare.apis.fl_snapshot import FLSnapshot, RunSnapshot
from nvflare.apis.state_persistor import StatePersistor
from nvflare.apis.storage import StorageSpec


class StorageStatePersistor(StatePersistor):
    def __init__(self, storage: StorageSpec, location: str):
        self.storage = storage
        self.location = location

        if not os.path.isabs(location):
            raise ValueError("snapshot location must be absolute path.")

    def save(self, snapshot: RunSnapshot) -> str:
        """Call to save the snapshot of the FL state to storage.
        Args:
            snapshot: FLSnapshot object
        Returns: storage location
        """
        path = os.path.join(self.location, snapshot.run_number)
        if snapshot.completed:
            self.storage.delete_object(path)
        else:
            self.storage.create_object(uri=path, data=pickle.dumps(snapshot), meta={}, overwrite_existing=True)

        return self.location

    def retrieve(self) -> FLSnapshot:
        """Call to load the persisted FL components snapshot from the persisted location.
        Args:
        Returns: retrieved Snapshot
        """
        all_items = self.storage.list_objects(path=self.location)
        fl_snapshot = FLSnapshot()
        for item in all_items:
            snapshot = pickle.loads(self.storage.get_data(item))
            fl_snapshot.add_snapshot(snapshot.run_number, snapshot)
        return fl_snapshot

    def retrieve_run(self, run_number: str) -> RunSnapshot:
        """Call to load the persisted RunSnapshot of a run_number from the persisted location.

        Args:
            run_number: run_number

        Returns: RunSnapshot of the run_number

        """
        path = os.path.join(self.location, run_number)
        snapshot = pickle.loads(self.storage.get_data(uri=path))
        return snapshot

    def delete(self, location: str):
        """To delete the FL component snapshot.
        Args:
            location: persist location
        Returns:
        """
        self.storage.delete_object(location)
