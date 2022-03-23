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

import datetime


class Study:
    def __init__(
        self,
        name: str,
        description: str,
        sites: [str],
        users: [str],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        reviewers=None,
    ):
        self.name = name
        self.description = description
        self.sites = sites
        self.users = users
        self.start_time = start_time
        self.end_time = end_time
        self.reviewers = reviewers
        self.create_time = None


class StudyManagerSpec(object):
    def create_study(self, study: Study) -> Study:
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
        pass

    def list_studies(self) -> [str]:
        """
        List names of all defined studies

        Returns: list of study names

        """
        pass

    def list_active_studies(self) -> [str]:
        """
        List names of all active studies (started but not ended)

        Returns: list of study names

        """
        pass

    def get_study(self, name: str) -> Study:
        """Get the Study object for the specified name.

        Args:
            name: unique name of the study

        Returns: the Study object

        """
        pass
