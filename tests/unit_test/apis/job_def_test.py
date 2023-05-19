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

from nvflare.apis.job_def import is_valid_job_id


class TestJobDef:
    def test_is_valid_job_id(self):
        assert not is_valid_job_id("site-1")
        assert is_valid_job_id("c2564481-536a-4548-8dfa-cf183a3652a1")
        assert is_valid_job_id("c2564481536a45488dfacf183a3652a1")
        assert not is_valid_job_id("c2564481536a45488dfacf183a3652a1ddd")
        assert not is_valid_job_id("c2564481-536a-4548-fdff-df183a3652a1")
