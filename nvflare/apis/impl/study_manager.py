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

import datetime
import json
import tempfile
from typing import List, Optional, Tuple

from nvflare.apis.storage import StorageSpec
from nvflare.apis.study_manager_spec import Study, StudyManagerSpec


def custom_json_encoder(obj):
    try:
        if isinstance(obj, datetime):
            return obj.isoformat()
        iterable = iter(obj)
    except TypeError:
        pass
    else:
        return list(iterable)


class StudyManager(StudyManagerSpec):
    def __init__(self, storage: StorageSpec):
        self._storage = storage
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._existing_studies = dict()

    def create_study(self, study: Study) -> Tuple[Optional[Study], str]:
        """Create the study object permanently

        The caller must have validated the sites and users of the study.

        Validate the study before saving:
        The name of the study must be unique;
        Sites and users must be defined;
        Start and end time must make sense.

        Args:
            study: the caller-provided study info

        Returns: updated study info (e.g. create_time is set)

        """
        if self.get_study(study.name):
            return None, "Unable to create duplicated study name."
        if not study.participating_clients:
            return None, "Study has no participating clients."
        if not study.contact:
            return None, "Study has no contact info."
        if not study.participating_admins:
            return None, "Study has no participating admins."
        if not study.start_date:
            return None, "Study has no start date."
        if not study.end_date:
            return None, "Study has no end date."
        study.create_time = datetime.datetime.utcnow()
        serialized_study = json.dumps(vars(study), default=custom_json_encoder).encode("utf-8")
        self._store.create_object(
            study.name, serialized_study, {"start_date": study.start_date, "end_date": study.end_date}
        )
        deserialized_study = json.loads(serialized_study.decode("utf-8"))
        deserialized_study["start_date"] = datetime.date.fromisoformat(deserialized_study["start_date"])
        deserialized_study["end_date"] = datetime.date.fromisoformat(deserialized_study["end_date"])
        stored_study = Study(**deserialized_study)  # for debugging purpose
        return study, ""

    def list_studies(self) -> List[str]:
        """
        List names of all defined studies

        Returns: list of study names

        """
        # either self._storage.list_object() or
        return list(self._existing_studies)

    def list_active_studies(self) -> List[str]:
        """
        List names of all active studies (started but not ended)

        Returns: list of study names

        """
        now = datetime.datetime.utcnow()
        active_studies = list()
        for k, v in self._existing_studies:
            if v["start_date"] >= now and v["end_date"] <= now:
                active_studies.append(k)
        return active_studies

    def get_study(self, name: str) -> Study:
        """Get the Study object for the specified name.

        Args:
            name: unique name of the study

        Returns: the Study object

        """
        serialized_study = self._storage.get_data(name)
        if serialized_study:
            deserialized_study = json.loads(serialized_study.decode("utf-8"))
        deserialized_study["start_date"] = datetime.date.fromisoformat(deserialized_study["start_date"])
        deserialized_study["end_date"] = datetime.date.fromisoformat(deserialized_study["end_date"])
        study = Study(**deserialized_study)
        return study
