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

from nvflare.apis.fl_snapshot import FLSnapshot, RunSnapshot
from nvflare.apis.state_persistor import StatePersistor
from nvflare.apis.storage import StorageSpec
from nvflare.fuel.utils import fobs


class StorageStatePersistor(StatePersistor):
    def __init__(self, storage: StorageSpec, uri_root: str):
        """Creates a StorageStatePersistor.

        Args:
            storage: StorageSpec object
            uri_root: where to store the states.
        """

        self.storage = storage
        self.uri_root = uri_root

    def save(self, snapshot: RunSnapshot) -> str:
        """Call to save the snapshot of the FL state to storage.

        Args:
            snapshot: RunSnapshot object

        Returns:
            storage location
        """
        path = os.path.join(self.uri_root, snapshot.job_id)
        if snapshot.completed:
            full_uri = self.storage.delete_object(path)
        else:
            full_uri = self.storage.create_object(uri=path, data=fobs.dumps(snapshot), meta={}, overwrite_existing=True)

        return full_uri

    def retrieve(self) -> FLSnapshot:
        """Call to load the persisted FL components snapshot from the persisted location.

        Returns:
            retrieved Snapshot
        """
        all_items = self.storage.list_objects(self.uri_root)
        fl_snapshot = FLSnapshot()
        for item in all_items:
            snapshot = fobs.loads(self.storage.get_data(item))
            fl_snapshot.add_snapshot(snapshot.job_id, snapshot)
        return fl_snapshot

    def retrieve_run(self, job_id: str) -> RunSnapshot:
        """Call to load the persisted RunSnapshot of a job from the persisted location.

        Args:
            job_id: job_id

        Returns:
            RunSnapshot of the job_id

        """
        path = os.path.join(self.uri_root, job_id)
        snapshot = fobs.loads(self.storage.get_data(uri=path))
        return snapshot

    def delete(self):
        """Deletes the FL snapshot."""

        all_items = self.storage.list_objects(self.uri_root)
        for item in all_items:
            self.storage.delete_object(item)

    def delete_run(self, job_id: str):
        """Deletes the RunSnapshot of a job.

        Args:
            job_id: job_id
        """
        path = os.path.join(self.uri_root, job_id)
        self.storage.delete_object(path)
