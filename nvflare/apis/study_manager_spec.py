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
from datetime import date, datetime
from typing import List, Optional, Tuple

from .fl_context import FLContext


class Study:
    def __init__(
        self,
        name: str,
        description: str,
        contact: str,
        participating_clients: List[str],
        participating_admins: List[str],
        start_date: date,
        end_date: date,
        reviewers=None,
        created_at: datetime = None,
    ):
        self.name = name
        self.description = description
        self.contact = contact
        self.participating_clients = participating_clients
        self.participating_admins = participating_admins
        self.start_date = start_date
        self.end_date = end_date
        self.reviewers = reviewers
        self.created_at = created_at


class StudyManagerSpec(ABC):
    @abstractmethod
    def add_study(self, study: Study, fl_ctx: FLContext) -> Tuple[Optional[Study], str]:
        """Add the study object permanently

        The caller must have validated the participating_clients and participating_admins of the study.

        Validate the study before saving:

            - The name of the study must be unique;
            - participating_clients and participating_admins must be defined;
            - Start and end date must make sense.

        Args:
            study: the caller-provided study info
            fl_ctx: FL context

        Returns:
            updated study info (e.g. created_at is set) and an emtpy string if successful
            None and an error message if the provided study is not valid
        """
        pass

    @abstractmethod
    def list_studies(self, fl_ctx: FLContext) -> List[str]:
        """List names of all defined studies

        Args:
            fl_ctx: FLContext

        Returns:
            A list of study names
        """
        pass

    @abstractmethod
    def list_active_studies(self, fl_ctx: FLContext) -> List[str]:
        """List names of all active studies (started but not ended)

        Args:
            fl_ctx: FLContext

        Returns:
            A list of study names
        """
        pass

    @abstractmethod
    def get_study(self, name: str, fl_ctx: FLContext) -> Study:
        """Get the Study object for the specified name.

        Args:
            name: unique name of the study
            fl_ctx: FLContext

        Returns:
            the Study object
        """
        pass
