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

import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as futures_thread

from nvflare.fuel.f3.streaming.stream_utils import CheckedExecutor, gen_stream_id


def generate_stream_ids(num_ids: int, result_queue: mp.Queue) -> None:
    """Worker function to generate stream IDs in a separate process.

    Args:
        num_ids: Number of stream IDs to generate
        result_queue: Queue to put the generated IDs
    """
    ids = []
    for _ in range(num_ids):
        ids.append(gen_stream_id())
    result_queue.put(ids)


class TestStreamUtils:
    """Test suite for stream_utils module"""

    def test_gen_stream_id_uniqueness_single_process(self):
        """Test that gen_stream_id generates unique IDs within a single process"""
        num_ids = 1000
        ids = [gen_stream_id() for _ in range(num_ids)]

        # Check for uniqueness
        assert len(ids) == len(set(ids)), "Generated IDs contain duplicates in single process"

        # Check that IDs are monotonically increasing
        assert ids == sorted(ids), "Generated IDs are not monotonically increasing"

    def test_gen_stream_id_returns_positive_int(self):
        """Test that gen_stream_id returns a positive integer"""
        stream_id = gen_stream_id()
        assert isinstance(stream_id, int), "Stream ID should be an integer"
        assert stream_id > 0, "Stream ID should be positive"

    def test_gen_stream_id_sequential_calls(self):
        """Test that sequential calls return increasing IDs"""
        id1 = gen_stream_id()
        id2 = gen_stream_id()
        id3 = gen_stream_id()

        assert id2 > id1, "Second ID should be greater than first"
        assert id3 > id2, "Third ID should be greater than second"
        assert id3 - id2 == id2 - id1 == 1, "IDs should increment by 1"

    def test_checked_executor_serializes_submit_with_shutdown(self, monkeypatch):
        """A submission admitted before shutdown must not race the base executor's stop."""
        executor = CheckedExecutor(max_workers=1, thread_name_prefix="checked_race")
        submit_entered = threading.Event()
        release_submit = threading.Event()
        submit_done = threading.Event()
        shutdown_started = threading.Event()
        shutdown_done = threading.Event()
        submit_errors = []
        real_submit = ThreadPoolExecutor.submit

        def paused_submit(pool, fn, *args, **kwargs):
            if pool is executor:
                submit_entered.set()
                assert release_submit.wait(2.0)
            return real_submit(pool, fn, *args, **kwargs)

        monkeypatch.setattr(ThreadPoolExecutor, "submit", paused_submit)

        def submit_work():
            try:
                future = executor.submit(lambda: "done")
                assert future.result(timeout=2.0) == "done"
            except BaseException as ex:
                submit_errors.append(ex)
            finally:
                submit_done.set()

        submit_thread = threading.Thread(target=submit_work)

        def shutdown_executor():
            shutdown_started.set()
            executor.shutdown(wait=True)
            shutdown_done.set()

        shutdown_thread = threading.Thread(target=shutdown_executor)
        submit_thread.start()
        assert submit_entered.wait(2.0)
        shutdown_thread.start()
        assert shutdown_started.wait(2.0)

        # Shutdown must wait until the already-admitted submit has crossed the
        # base executor boundary; the previous check-then-submit race did not.
        assert not shutdown_done.wait(0.05)
        release_submit.set()

        assert submit_done.wait(2.0)
        assert shutdown_done.wait(2.0)
        submit_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=2.0)
        assert submit_errors == []
        assert executor.submit(lambda: None) is None

    def test_checked_executor_ignores_base_executor_already_shut_down(self):
        """A base-level shutdown cannot leak RuntimeError into F3 teardown workers."""
        executor = CheckedExecutor(max_workers=1, thread_name_prefix="checked_base_stop")

        ThreadPoolExecutor.shutdown(executor, wait=True)

        assert executor.submit(lambda: None) is None
        assert executor.stopped is True

    def test_checked_executor_ignores_interpreter_executor_shutdown(self, monkeypatch):
        executor = CheckedExecutor(max_workers=1, thread_name_prefix="checked_interpreter_stop")
        monkeypatch.setattr(futures_thread, "_shutdown", True)

        assert executor.submit(lambda: None) is None
        assert executor.stopped is True
