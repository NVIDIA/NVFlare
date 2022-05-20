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

from .fl_snapshot import FLSnapshot, RunSnapshot


class StatePersistor:
    def save(self, snapshot: RunSnapshot) -> str:
        """Call to save the snapshot of the FL state to storage.

        Args:
            snapshot: RunSnapshot object

        Returns:
            Storage location.
        """
        pass

    def retrieve(self) -> FLSnapshot:
        """Call to load the persisted FL components snapshot from the persisted location.

        Args:

        Returns: FLSnapshot

        """
        pass

    def retrieve_run(self, run_number: str) -> RunSnapshot:
        """Call to load the persisted RunSnapshot of a run_number from the persisted location.

        Args:
            run_number: run_number

        Returns: RunSnapshot of the run_number

        """
        pass

    def delete(self, location: str):
        """Delete the FL component snapshot.

        Args:
            location: persist location

        Returns:

        """
        pass
