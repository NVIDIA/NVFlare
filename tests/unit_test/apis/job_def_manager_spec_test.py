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

from nvflare.apis.job_def_manager_spec import JobDefManagerSpec


def test_submit_token_extension_methods_are_not_abstract():
    extension_methods = {
        "get_job_content_hash",
        "get_submit_record",
        "new_submit_record",
        "create_submit_record",
        "update_submit_record",
        "get_job_by_submit_token",
    }

    assert extension_methods.isdisjoint(JobDefManagerSpec.__abstractmethods__)


def test_new_submit_record_default_is_instance_method():
    assert not isinstance(JobDefManagerSpec.__dict__["new_submit_record"], staticmethod)


def test_submit_token_extension_defaults_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        JobDefManagerSpec.get_job_content_hash(object(), b"job")
    with pytest.raises(NotImplementedError):
        JobDefManagerSpec.get_submit_record(object(), "study", {}, "token", None)
    with pytest.raises(NotImplementedError):
        JobDefManagerSpec.new_submit_record(object(), "study", {}, "token", "hash")
    with pytest.raises(NotImplementedError):
        JobDefManagerSpec.create_submit_record(object(), {}, None)
    with pytest.raises(NotImplementedError):
        JobDefManagerSpec.update_submit_record(object(), {}, None)
    with pytest.raises(NotImplementedError):
        JobDefManagerSpec.get_job_by_submit_token(object(), "study", {}, "token", None)
