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

from nvflare import cli as cli_mod


def test_auth_hint_for_unknown_study():
    assert cli_mod._auth_hint_from_detail("unknown study 'cancer_research'") == (
        "Add the study under 'studies:' in project.yml with api_version: 4, reprovision, redeploy or restart the server, then try again."
    )


def test_auth_hint_for_missing_study_mapping():
    assert cli_mod._auth_hint_from_detail("user 'admin@nvidia.com' is not mapped to study 'cancer_research'") == (
        "Add this user under the study's admins mapping in project.yml, reprovision, redeploy or restart the server, then try again."
    )


def test_auth_hint_for_invalid_study_name():
    assert cli_mod._auth_hint_from_detail("invalid study name 'bad study'") == (
        "Use a valid study name in project.yml, reprovision, redeploy or restart the server, then try again."
    )


def test_auth_hint_defaults_to_credentials():
    assert cli_mod._auth_hint_from_detail("Incorrect user name or password") == "Check startup kit credentials."
