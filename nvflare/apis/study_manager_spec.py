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

from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime

from .fl_context import FLContext


class Study:
    def __init__(
        self,
        name: str,
        description: str,
        contact: str,
        participating_clients: List[str],
        participating_admins: List[str],
        start_date: datetime.date,
        end_date: datetime.date,
        reviewers=None,
    ):
        self.name = name
        self.description = description
        self.contact = contact
        self.participating_clients = participating_clients
        self.participating_admins = participating_admins
        self.start_date = start_date
        self.end_date = end_date
        self.reviewers = reviewers
        self.created_at = datetime.utcnow().isoformat()


class StudyManagerSpec(ABC):
    @abstractmethod
    def add_study(self, study: Study, fl_ctx: FLContext) -> Study:
        """Add the study object permanently

        The caller must have validated the participating_clients and participating_admins of the study.

        Validate the study before saving:
        The name of the study must be unique;
        participating_clients and participating_admins must be defined;
        Start and end date must make sense.

        Args:
            study: the caller-provided study info

        Returns: updated study info (e.g. create_time is set)

        """
        pass

    @abstractmethod
    def list_studies(self, fl_ctx: FLContext) -> List[str]:
        """
        List names of all defined studies

        Returns: list of study names

        """
        pass

    @abstractmethod
    def list_active_studies(self, fl_ctx: FLContext) -> List[str]:
        """
        List names of all active studies (started but not ended)

        Returns: list of study names

        """
        pass

    @abstractmethod
    def get_study(self, name: str, fl_ctx: FLContext) -> Study:
        """Get the Study object for the specified name.

        Args:
            name: unique name of the study

        Returns: the Study object

        """
        pass
