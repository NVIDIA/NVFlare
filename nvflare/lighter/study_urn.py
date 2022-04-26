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

from enum import Enum


class StudyUrn(str, Enum):
    """Constants used for key names in for Study information."""

    NAME = "urn:nvidia:nvflare:study:name"
    DESCRIPTION = "urn:nvidia:nvflare:study:description"
    CONTACT = "urn:nvidia:nvflare:study:contact"
    ADMINS = "urn:nvidia:nvflare:study:admins"
    CLIENTS = "urn:nvidia:nvflare:study:clients"
    START_TIME = "urn:nvidia:nvflare:study:start_time"
    END_TIME = "urn:nvidia:nvflare:study:end_time"
    STUDIES = "urn:nvidia:nvflare:study:studies"
