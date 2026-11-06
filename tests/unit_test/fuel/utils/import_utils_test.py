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

import sys
import types

import pytest

from nvflare.fuel.utils.import_utils import LazyImportError, optional_import

FAKE_MODULE = "_nvflare_fake_versioned_module"


@pytest.fixture
def fake_module(monkeypatch):
    def _make(version):
        mod = types.ModuleType(FAKE_MODULE)
        mod.__version__ = version
        mod.value = 42
        monkeypatch.setitem(sys.modules, FAKE_MODULE, mod)
        return mod

    return _make


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

    @pytest.mark.parametrize(
        "installed, op, required, expected",
        [
            ("2.4.6", ">=", "2.4.0", True),
            ("2.4.6", ">=", "2.4.6", True),
            ("2.4.6", ">=", "2.4.7", False),
            ("2.4.6", "==", "2.4.6", True),
            ("2.4.6", "==", "2.4", False),
            ("2.4.6", ">", "2.4.1", True),
            ("2.4.6", ">", "2.4.6", False),
            ("2.4.6", "<", "2.4.7", True),
            ("2.4.6", "<=", "2.4.5", False),
            ("2.4", "==", "2.4.0", True),
            ("2.4.0", ">=", "2.4", True),
            ("2.4", "<", "2.4.1", True),
            ("2.4.6rc1", ">=", "2.4.6", True),
            ("2.1.0+cu121", "==", "2.1.0", True),
            ("2.1.0+cu121", ">=", "2.1.1", False),
            ("1.26.0.dev0", "<", "2", True),
            ("1!2.0", ">=", "1!3.0", False),
            ("1!2.0", ">=", "3.0", True),
            ("2.0", "<", "1!1.0", True),
            ("1!2.0", "==", "1!2", True),
        ],
    )
    def test_version_uses_all_components(self, fake_module, installed, op, required, expected):
        fake_module(installed)
        mod, flag = optional_import(FAKE_MODULE, op, required)
        assert flag is expected
        if expected:
            assert mod.value == 42
        else:
            with pytest.raises(LazyImportError):
                mod.value

    @pytest.mark.parametrize(
        "installed, required",
        [
            ("2.4.6", "2.4.6garbage"),
            ("2.4.6garbage", "2.4.6"),
            ("2.4.6", "2.4.6 7"),
            ("1!2.0x", "1!2.0"),
            ("2.4.6", "!2.0"),
        ],
    )
    def test_invalid_version_raises(self, fake_module, installed, required):
        fake_module(installed)
        with pytest.raises(ValueError, match="cannot parse version"):
            optional_import(FAKE_MODULE, ">=", required)

    def test_installed_numpy_version_satisfies_itself(self):
        import numpy

        _, flag = optional_import("numpy", "==", numpy.__version__)
        assert flag is True
        _, flag = optional_import("numpy", ">=", numpy.__version__)
        assert flag is True

    def test_version_error_shows_full_installed_version(self, fake_module):
        fake_module("2.4.6")
        mod, flag = optional_import(FAKE_MODULE, ">=", "2.5.0")
        assert flag is False
        with pytest.raises(LazyImportError) as exc_info:
            mod.value
        assert f"requires '{FAKE_MODULE}>=2.5.0', current '{FAKE_MODULE}==2.4.6'." in str(exc_info.value)
