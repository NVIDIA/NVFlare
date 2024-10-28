# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import inspect

from nvflare.fuel.utils.class_utils import get_component_init_parameters
from nvflare.private.fed.server.client_manager import ClientManager


class A:
    def __init__(self, name: str = None):
        self.name = name


class B(A):
    def __init__(self, type: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = type


class TestClassUtils:
    def test_get_component_init_parameters(self):
        manager = ClientManager(project_name="Sample project", min_num_clients=1)
        constructor = ClientManager.__init__
        expected_parameters = inspect.signature(constructor).parameters
        parameters = get_component_init_parameters(manager)

        assert parameters == expected_parameters

    def test_nested_init_parameters(self):
        expected_parameters = {}
        constructor = B.__init__
        b = B(name="name", type="type")
        expected_parameters.update(inspect.signature(constructor).parameters)
        constructor = A.__init__
        expected_parameters.update(inspect.signature(constructor).parameters)
        parameters = get_component_init_parameters(b)

        assert parameters == expected_parameters
