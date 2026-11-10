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


@pytest.fixture
def isolated_fobs():
    # Lightweight suites also load this conftest; import runtime dependencies
    # only when a worker/artifact test requests FOBS isolation.
    from nvflare.apis.utils.decomposers import flare_decomposers
    from nvflare.app_common.decomposers import common_decomposers
    from nvflare.fuel.utils.fobs import fobs as fobs_registry
    from nvflare.fuel.utils.fobs.builtin_decomposers import BUILTIN_TYPES
    from nvflare.private.fed.utils.decomposers import private_decomposers
    from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize

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


@pytest.fixture(scope="session")
def dashboard_app():
    """Share an isolated Dashboard database without eagerly importing optional dependencies."""
    import tempfile

    from nvflare.dashboard.application import db, init_app
    from nvflare.dashboard.config import Config
    from tests.dashboard_test_utils import TEST_PW, TEST_USER

    with tempfile.TemporaryDirectory(prefix="nvflare-dashboard-") as folder, pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", f"sqlite:///{folder}/db.sqlite")
        patch.setenv("NVFL_WEB_ROOT", folder)
        patch.setenv("NVFL_CREDENTIAL", f"{TEST_USER}:{TEST_PW}:nvidia")
        patch.setattr(Config, "SQLALCHEMY_DATABASE_URI", f"sqlite:///{folder}/db.sqlite")
        application = init_app()
        application.config.update(TESTING=True)
        try:
            yield application
        finally:
            with application.app_context():
                db.session.remove()
                db.engine.dispose()
