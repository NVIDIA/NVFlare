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

from unittest.mock import patch

import pytest

from nvflare.apis.app_validation import AppValidationKey, AppValidator
from nvflare.fuel.sec.authz import AuthorizationService
from nvflare.private.fed.utils.app_authz import AppAuthzService
from nvflare.security.security import EmptyAuthorizer


class _MockAppValidator(AppValidator):
    """Mock AppValidator for testing."""

    def __init__(self, validate_return=("", {})):
        self.validate_return = validate_return

    def validate(self, app_folder: str):
        return self.validate_return


@pytest.fixture(autouse=True)
def setup_authz():
    """Initialize AuthorizationService with EmptyAuthorizer for all tests."""
    AuthorizationService.initialize(EmptyAuthorizer())
    yield


class TestAppAuthzService:
    """Test AppAuthzService authorization logic for Flower predeployed mode."""

    def test_authorize_flower_predeployed_granted(self, tmp_path):
        """Predeployed mode with granted right -> succeeds."""
        app_path = str(tmp_path / "app")
        (tmp_path / "app").mkdir()

        mock_validator = _MockAppValidator(validate_return=("", {}))

        AppAuthzService.initialize(mock_validator)

        with patch.object(AuthorizationService, "authorize", return_value=(True, "")):
            authorized, error = AppAuthzService.authorize(
                app_path=app_path,
                submitter_name="test_user",
                submitter_org="test_org",
                submitter_role="org_admin",
                job_meta={AppValidationKey.FLOWER_PREDEPLOYED: True},
            )

        assert authorized
        assert error == ""

    def test_authorize_flower_predeployed_denied(self, tmp_path):
        """Predeployed mode with denied right -> fails with clear error."""
        app_path = str(tmp_path / "app")
        (tmp_path / "app").mkdir()

        mock_validator = _MockAppValidator(validate_return=("", {}))

        AppAuthzService.initialize(mock_validator)

        with patch.object(AuthorizationService, "authorize", return_value=(False, "not permitted")):
            authorized, error = AppAuthzService.authorize(
                app_path=app_path,
                submitter_name="test_user",
                submitter_org="test_org",
                submitter_role="org_admin",
                job_meta={AppValidationKey.FLOWER_PREDEPLOYED: True},
            )

        assert not authorized
        assert "Server-predeployed Flower app mode is not permitted" in error
        assert "server-predeployed-flwr" in error

    def test_authorize_flower_predeployed_no_job_meta(self, tmp_path):
        """Predeployed mode without job_meta -> skips check (backward compatibility)."""
        app_path = str(tmp_path / "app")
        (tmp_path / "app").mkdir()

        mock_validator = _MockAppValidator(validate_return=("", {}))

        AppAuthzService.initialize(mock_validator)

        with patch.object(AuthorizationService, "authorize") as mock_auth:
            authorized, error = AppAuthzService.authorize(
                app_path=app_path,
                submitter_name="test_user",
                submitter_org="test_org",
                submitter_role="org_admin",
            )

        assert authorized
        assert error == ""

    def test_authorize_byoc_still_works(self, tmp_path):
        """Existing BYOC functionality unchanged."""
        app_path = str(tmp_path / "app")
        (tmp_path / "app").mkdir()

        mock_validator = _MockAppValidator(validate_return=("", {AppValidationKey.BYOC: True}))

        AppAuthzService.initialize(mock_validator)

        with patch.object(AuthorizationService, "authorize", return_value=(True, "")):
            authorized, error = AppAuthzService.authorize(
                app_path=app_path,
                submitter_name="test_user",
                submitter_org="test_org",
                submitter_role="org_admin",
                job_meta={},
            )

        assert authorized
        assert error == ""

    def test_authorize_both_flags(self, tmp_path):
        """Both BYOC and predeployed flags -> both checks run."""
        app_path = str(tmp_path / "app")
        (tmp_path / "app").mkdir()

        mock_validator = _MockAppValidator(validate_return=("", {AppValidationKey.BYOC: True}))

        AppAuthzService.initialize(mock_validator)

        with patch.object(AuthorizationService, "authorize", return_value=(True, "")) as mock_auth:
            authorized, error = AppAuthzService.authorize(
                app_path=app_path,
                submitter_name="test_user",
                submitter_org="test_org",
                submitter_role="org_admin",
                job_meta={AppValidationKey.FLOWER_PREDEPLOYED: True},
            )

        assert authorized
        assert mock_auth.call_count == 2
        assert error == ""

    def test_authorize_neither_flag(self, tmp_path):
        """No custom code, no predeployed flag -> succeeds without auth check."""
        app_path = str(tmp_path / "app")
        (tmp_path / "app").mkdir()

        mock_validator = _MockAppValidator(validate_return=("", {}))

        AppAuthzService.initialize(mock_validator)

        with patch.object(AuthorizationService, "authorize") as mock_auth:
            authorized, error = AppAuthzService.authorize(
                app_path=app_path,
                submitter_name="test_user",
                submitter_org="test_org",
                submitter_role="org_admin",
                job_meta={},
            )

        assert authorized
        assert error == ""
        mock_auth.assert_not_called()
