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

import pickle
import os

from nvflare.apis.state_persistor import FLSnapshot, StatePersistor


class SnapshotFilePersistor(StatePersistor):

    def __init__(self, location) -> None:
        super().__init__()
        self.location = location

        if not os.path.isabs(location):
            raise ValueError("snapshot location must be absolute path.")

    def save(self, snapshot: FLSnapshot) -> str:
        with open(self.location, "wb") as f:
            f.write(pickle.dumps(snapshot))
        return self.location

    def retrieve(self) -> FLSnapshot:
        snapshot = None
        if os.path.exists(self.location):
            with open(self.location, "rb") as f:
                data = f.read()
                snapshot = pickle.loads(data)
        return snapshot

    def delete(self, location: str):
        os.remove(location)
