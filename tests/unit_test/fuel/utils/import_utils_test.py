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

import pytest

from nvflare.fuel.utils.import_utils import LazyImportError, optional_import


class TestOptionalImport:
    def test_lazy_import(self):
        np, flag = optional_import("numpy")
        assert flag is True

        np, flag = optional_import(module="numpy", op=">=", version="1.0.0")
        assert flag is True

        np, flag = optional_import("numpy", ">=", "100.0.0")
        assert flag is False
        with pytest.raises(LazyImportError):
            print(np.will_faill)

        # numpy is 1.22
        # np, flag = optional_import("numpy", "==", "1.22")
        # assert flag == True

        the_module, flag = optional_import("unknown_module")
        with pytest.raises(LazyImportError):
            print(the_module.method)  # trying to access a module which is not imported

        torch, flag = optional_import("torch", "==", "42")
        with pytest.raises(LazyImportError):
            print(torch.nn)  # trying to access a module for which there isn't a proper version imported

        # if you have torch installed. uncomment this line
        # conv, flag = optional_import(module="torch.nn.functional", name="conv1d")
        # print(conv)
        # assert flag == True

        with pytest.raises(LazyImportError):
            conv, flag = optional_import(module="torch", op=">=", version="42")
            # trying to use a function from the not successfully imported module (due to unmatched version)
            print(conv())
        with pytest.raises(LazyImportError):
            conv, flag = optional_import(module="torch.nn.functional", op=">=", version="42", name="conv1d")
            # trying to use a function from the not successfully imported module (due to unmatched version)
            print(conv())
