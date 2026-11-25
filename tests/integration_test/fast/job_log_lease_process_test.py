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

"""Check that a job-log lease protects files from a separate process."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from nvflare.app_common.logging.log_staging import ClientLogStaging, lock_file
from tests.process_test_utils import run_owned_process
from tests.timing_utils import PROCESS_STARTUP_TIMEOUT


def test_separate_process_preserves_leased_job_logs_until_close(tmp_path):
    log_root = str(tmp_path)
    staging = ClientLogStaging(SimpleNamespace(get_log_root=lambda: log_root))
    job_id = "job-1"
    site_logs = Path(staging.root) / job_id / "site-1"
    site_logs.mkdir(parents=True)
    text_log = site_logs / "log.txt"
    json_log = site_logs / "log.json"

    with text_log.open("wb") as first, json_log.open("wb") as second:
        for stream, contents in ((first, b"text log"), (second, b"json log")):
            assert lock_file(stream.fileno())
            stream.write(contents)
            stream.flush()
        with staging.locked():
            staging.prepare_delete(job_id)
            staging.mark_deleted(job_id)
        result = run_owned_process(
            [
                sys.executable,
                "-c",
                "import sys\nfrom types import SimpleNamespace\n"
                "from nvflare.app_common.logging.log_staging import ClientLogStaging\n"
                "staging = ClientLogStaging(SimpleNamespace(get_log_root=lambda: sys.argv[1]))\n"
                "with staging.locked():\n    staging.remove_closed_job_logs(sys.argv[2])\n",
                log_root,
                job_id,
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=os.environ.copy(),
            timeout=PROCESS_STARTUP_TIMEOUT,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert text_log.read_bytes() == b"text log"
        assert json_log.read_bytes() == b"json log"

    with staging.locked():
        staging.remove_closed_job_logs(job_id)
    assert not site_logs.parent.exists()
