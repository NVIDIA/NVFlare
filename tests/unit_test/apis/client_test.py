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

from nvflare.apis.client import Client, ClientDictKey, from_dict


def test_client_site_config_to_dict_from_dict():
    site_config = {
        "format_version": 1,
        "resources": {"memory_gb": 128},
        "labels": {"region": "us-east"},
        "capabilities": ["he", "psi"],
    }

    client = Client(name="site-1", token="token")
    client.set_site_config(site_config)

    site_config["labels"]["region"] = "mutated"
    client_dict = client.to_dict()

    assert client_dict[ClientDictKey.SITE_CONFIG]["labels"]["region"] == "us-east"

    client_dict[ClientDictKey.SITE_CONFIG]["labels"]["region"] = "from-dict"
    restored = from_dict(client_dict)
    client_dict[ClientDictKey.SITE_CONFIG]["labels"]["region"] = "mutated-again"

    assert restored.get_site_config()["labels"]["region"] == "from-dict"
