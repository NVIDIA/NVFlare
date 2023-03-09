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

from nvflare.app_common.np.np_model_locator import NPModelLocator
from tests.unit_test.fuel.utils.mock_component_builder import MockComponentBuilder


class MyComponent:
    def __init__(self, model):
        self.mode = model


class TestComponentBuilder:

    def test_empty_dict(self):
        builder = MockComponentBuilder({})
        b = builder.build_component2()
        assert (b is None)

    def test_component(self):
        config = {
            "id": "id",
            "path": "nvflare.app_common.np.np_model_locator.NPModelLocator",
            "args": {
            }
        }
        builder = MockComponentBuilder(config)

        assert isinstance(config, dict)
        b = builder.build_component2()
        assert isinstance(b, NPModelLocator)
    #
    # def test_embedded_component(self):
    #     config = {
    #         "id": "id",
    #         "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
    #         "args": {
    #             "model": {
    #                 "path": "nvflare.app_common.np.np_model_locator.NPModelLocator",
    #                 "args": {
    #                 }
    #             }
    #         }
    #     }
    #     builder = MockComponentBuilder(config)
    #     assert isinstance(config, dict)
    #     b = builder.build_component2()
    #     assert isinstance(b, MyComponent)
    #
