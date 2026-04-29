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

import io
from zipfile import ZipFile

from nvflare.apis.utils.job_submit_token import canonical_job_content_hash
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE, NVFLARE_SUBMITTER_CRT_FILE


def _zip_bytes(files):
    output = io.BytesIO()
    with ZipFile(output, "w") as zip_file:
        for name, content in files.items():
            zip_file.writestr(name, content)
    return output.getvalue()


def test_canonical_job_content_hash_ignores_signing_artifacts():
    base = _zip_bytes(
        {
            "hello/meta.json": "{}",
            "hello/app/config/config_fed_server.json": "{}",
        }
    )
    signed = _zip_bytes(
        {
            "hello/meta.json": "{}",
            "hello/app/config/config_fed_server.json": "{}",
            f"hello/{NVFLARE_SIG_FILE}": '{"signature": "volatile"}',
            f"hello/{NVFLARE_SUBMITTER_CRT_FILE}": "volatile cert",
        }
    )

    assert canonical_job_content_hash(base) == canonical_job_content_hash(signed)
