# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_context import FLContext
from nvflare.private.defs import ClientRegMsgKey
from nvflare.private.fed.client.communicator import Communicator


def test_get_site_config_for_registration_from_loaded_client_config():
    site_config = {"labels": {"region": "us-east"}}
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: site_config})

    assert communicator._get_site_config_for_registration(FLContext()) == site_config


def test_get_site_config_for_registration_ignores_non_dict_config():
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: ["bad"]})

    assert communicator._get_site_config_for_registration(FLContext()) is None
