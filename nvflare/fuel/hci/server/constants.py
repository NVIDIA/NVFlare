# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


class ConnProps(object):
    """Constants for connection properties."""

    EVENT_ID = "_eventId"
    USER_NAME = "_userName"
    USER_ORG = "_userOrg"
    USER_ROLE = "_userRole"
    SUBMITTER_NAME = "_submitterName"
    SUBMITTER_ORG = "_submitterOrg"
    SUBMITTER_ROLE = "_submitterRole"
    TOKEN = "_sessionToken"
    SESSION = "_session"
    CLIENT_IDENTITY = "_clientIdentity"
    CA_CERT = "_caCert"
    UPLOAD_DIR = "_uploadDir"
    DOWNLOAD_DIR = "_downloadDir"
    DOWNLOAD_JOB_URL = "_downloadJobUrl"

    CMD_ENTRY = "_cmdEntry"
    JOB_DATA = "_jobData"
    JOB_META = "_jobMeta"
