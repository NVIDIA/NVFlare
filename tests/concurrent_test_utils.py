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

"""Concurrency scenarios run under a parent-process deadlock watchdog."""

from unittest.mock import MagicMock

from tests.timing_utils import ThreadGroup, join_thread


def concurrent_run_results():
    from nvflare.recipe.run import Run

    environment = MagicMock()
    environment.get_job_result.return_value = "/tmp/workspace/test_job_123"
    environment.get_job_status.return_value = "FINISHED"
    run = Run(exec_env=environment, job_id="test_job_123")
    results = []
    with ThreadGroup() as workers:
        for _ in range(5):
            workers.start(lambda: results.append(run.get_result()))
        for thread in workers.threads:
            join_thread(thread)
    assert results == ["/tmp/workspace/test_job_123"] * 5
    assert environment.stop.call_count == 1


def concurrent_downloadable_release():
    import numpy as np

    from nvflare.app_common.np.np_downloader import ArrayDownloadable

    arrays = {f"key_{i}": np.ones((256, 256), dtype=np.float32) for i in range(4)}
    downloadable = ArrayDownloadable(arrays=arrays, max_chunk_size=2 * 1024 * 1024)
    with ThreadGroup() as workers:
        for _ in range(10):
            workers.start(downloadable.release)
        for thread in workers.threads:
            join_thread(thread)
    assert downloadable.base_obj is None
