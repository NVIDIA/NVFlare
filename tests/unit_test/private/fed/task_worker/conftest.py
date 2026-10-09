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

import pytest

from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.utils.fobs import fobs as fobs_registry
from nvflare.fuel.utils.fobs.builtin_decomposers import BUILTIN_TYPES
from nvflare.private.fed.utils.decomposers import private_decomposers
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


@pytest.fixture(autouse=True)
def _initialize_fobs():
    # Earlier tests may reset FOBS and auto-register a generic Shareable
    # decomposer. register() will not replace it, so it can disagree with the
    # DictDecomposer used by a fresh worker process. Isolate the registry and
    # registration guards together, restoring the previous objects at teardown.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fobs_registry, "_decomposers", {})
        patch.setattr(fobs_registry, "_dot_handlers", {})
        patch.setattr(fobs_registry, "_type_name_whitelist", set(BUILTIN_TYPES))
        patch.setattr(fobs_registry, "_decomposers_registered", False)
        patch.setattr(fobs_registry, "_enum_auto_registration", True)
        patch.setattr(fobs_registry, "_data_auto_registration", True)
        for module in (flare_decomposers, common_decomposers, private_decomposers):
            patch.setattr(module.register, "registered", False)
        nvflare_fobs_initialize()
        yield
