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
from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.dh_psi.dh_psi_workflow import DhPSIWorkFlow


class TestDhPSIWorkflow:
    def test_get_ordered_sites(self):
        wf = DhPSIWorkFlow()
        data_1 = {PSIConst.ITEMS_SIZE: 1000}
        data_2 = {PSIConst.ITEMS_SIZE: 500}
        data_3 = {PSIConst.ITEMS_SIZE: 667}
        dxo_1 = DXO(data_kind=DataKind.PSI, data=data_1)
        dxo_2 = DXO(data_kind=DataKind.PSI, data=data_2)
        dxo_3 = DXO(data_kind=DataKind.PSI, data=data_3)
        results = {"site-1": dxo_1, "site-2": dxo_2, "site-3": dxo_3}
        ordered_sites = wf.get_ordered_sites(results=results)

        assert ordered_sites[0].size <= ordered_sites[1].size
        assert ordered_sites[1].size <= ordered_sites[2].size
