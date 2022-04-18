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

import json
import logging
from datetime import date, datetime
from typing import List, Optional, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.apis.study_manager_spec import Study, StudyManagerSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.storage import StorageSpec


def custom_json_encoder(obj):
    try:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        iterable = iter(obj)
    except TypeError:
        pass
    else:
        return list(iterable)


class StudyManager(StudyManagerSpec):
    STORAGE_KEY = "std_mgr"

    def __init__(self, study_store_id: str):
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)
        self._study_store_id = study_store_id
        try:
            self._existing_studies = json.loads(self._storage.get_data(StudyManager.STORAGE_KEY).decode("utf-8"))
        except BaseException:
            self._existing_studies = list()

    def _get_store(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ServerEngineSpec):
            raise TypeError(f"engine should be of type ServerEngineSpec, but got {type(engine)}")
        store = engine.get_component(self._study_store_id)
        if not isinstance(store, StorageSpec):
            raise TypeError(f"engine should have a job store component of type StorageSpec, but got {type(store)}")
        return store

    def add_study(self, study: Study, fl_ctx: FLContext) -> Tuple[Optional[Study], str]:
        """Add the study object permanently

        The caller must have validated the participating_clients and participating_admins of the study.

        Validate the study before saving:

            - The name of the study must be unique;
            - participating_clients and participating_admins must be defined;
            - Start and end date must make sense.

        Args:
            study: the caller-provided study info
            fl_ctx: FLContext

        Returns:
            updated study info (e.g. created_at is set) and an emtpy string if successful
            None and an error message if the provided study is not valid
        """
        if study.name in self._existing_studies:
            return None, "Unable to add duplicated study name."
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
        if study.end_date < study.start_date:
            return (
                None,
                f"Expect end_date later than start_date.  Got start_date={study.start_date} and end_data={study.end_date}",
            )
        study.created_at = datetime.utcnow()
        serialized_study = self._study_to_bytes(study)
        try:
            store = self._get_store(fl_ctx)
            store.create_object(
                uri=study.name,
                data=serialized_study,
                meta={"start_date": study.start_date, "end_date": study.end_date},
                overwrite_existing=True,
            )
            store.create_object(
                uri=StudyManager.STORAGE_KEY,
                data=json.dumps(self._existing_studies + [study.name]).encode("utf-8"),
                meta={},
                overwrite_existing=True,
            )
        except BaseException as e:
            self.logger.warning(f"Unable to add study into storage due to {e}")
            return None, f"Unable to add study into storage due to {e}"
        self._existing_studies.append(study.name)
        return study, ""

    @staticmethod
    def _study_from_bytes(bytes_study):
        try:
            deserialized_study = json.loads(bytes_study.decode("utf-8"))
            deserialized_study["start_date"] = date.fromisoformat(deserialized_study.pop("start_date"))
            deserialized_study["end_date"] = date.fromisoformat(deserialized_study.pop("end_date"))
            deserialized_study["created_at"] = datetime.fromisoformat(deserialized_study.pop("created_at"))
            return Study(**deserialized_study)
        except BaseException:
            return None

    @staticmethod
    def _study_to_bytes(study):
        try:
            serialized_study = json.dumps(vars(study), default=custom_json_encoder).encode("utf-8")
            return serialized_study
        except BaseException:
            return None

    def list_studies(self, fl_ctx: FLContext) -> List[str]:
        """List names of all defined studies

        Args:
            fl_ctx: FLContext

        Returns:
            A list of study names
        """
        # either self._storage.list_object() or
        return self._existing_studies

    def list_active_studies(self, fl_ctx: FLContext) -> List[str]:
        """List names of all active studies (started but not ended)

        Args:
            fl_ctx: FLContext

        Returns:
            A list of study names
        """
        today = date.today()
        active_studies = list()
        for st in self._existing_studies:
            try:
                meta = self._storage.get_meta(st)
                if meta["start_date"] > today:
                    continue
                if meta["end_date"] <= today:
                    continue
                active_studies.append(st)
            except BaseException as e:
                self.logger.warning(
                    f"Inconsistency between internal persisted states.  Expect {st} with meta in storage but got {e}."
                )
        return active_studies

    def get_study(self, name: str, fl_ctx: FLContext) -> Optional[Study]:
        """Get the Study object for the specified name.

        Args:
            name: unique name of the study
            fl_ctx: FLContext

        Returns:
            the Study object
        """
        try:
            store = self._get_store(fl_ctx)
            serialized_study = store.get_data(name)
        except RuntimeError as e:
            self.logger.warning(f"Unable to load study {name} from storage.")
            return None
        study = self._study_from_bytes(serialized_study)
        return study
