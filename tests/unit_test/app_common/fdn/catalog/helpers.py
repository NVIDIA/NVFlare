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

from nvflare.app_common.fdn.catalog.common.limits import ReadLimits

LIMITS = ReadLimits(max_duration_s=60, max_calls=100, max_records=1000)


def contract_entry(dataset="clinical_records"):
    contract = {
        "apiVersion": "v3.2.0",
        "kind": "DataContract",
        "version": "1",
        "id": "test",
        "schema": [{"name": "visits", "properties": [{"name": "id"}]}],
    }
    return {"dataset": dataset, "contract": contract}
