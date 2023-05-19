# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# need to be consistent with provision
RESOURCE_CONFIG = "resources.json"
DEFAULT_RESOURCE_CONFIG = "resources.json.default"
SERVER_NVF_CONFIG = "fed_server.json"
CLIENT_NVF_CONFIG = "fed_client.json"


FILE_STORAGE = "nvflare.app_common.storages.filesystem_storage.FilesystemStorage"

SERVER_SCRIPT = "nvflare.private.fed.app.server.server_train"
CLIENT_SCRIPT = "nvflare.private.fed.app.client.client_train"


# provision
PROVISION_SCRIPT = "nvflare.cli provision"

# preflight check
PREFLIGHT_CHECK_SCRIPT = "nvflare.cli preflight_check"
