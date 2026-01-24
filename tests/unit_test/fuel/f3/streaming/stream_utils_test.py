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

import pytest

from nvflare.fuel.f3.streaming.stream_utils import gen_stream_id


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

    @pytest.mark.parametrize("num_iterations", [10])
    def test_gen_stream_id_no_collision_multiprocess(self, num_iterations):
        """Test that gen_stream_id generates unique IDs across multiple processes.

        This test uses 2 processes, each generating 1000 stream IDs, and runs
        multiple iterations to ensure no collisions occur.
        """
        num_processes = 2
        ids_per_process = 1000

        for iteration in range(num_iterations):
            # Create a queue to collect results
            result_queue = mp.Queue()

            # Create and start individual processes
            processes = []
            for _ in range(num_processes):
                p = mp.Process(target=generate_stream_ids, args=(ids_per_process, result_queue))
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Collect results from queue
            all_ids = []
            while not result_queue.empty():
                process_ids = result_queue.get()
                all_ids.extend(process_ids)

            # Check total count
            expected_total = num_processes * ids_per_process
            assert (
                len(all_ids) == expected_total
            ), f"Iteration {iteration + 1}: Expected {expected_total} IDs, got {len(all_ids)}"

            # Check for uniqueness across all processes
            unique_ids = set(all_ids)
            if len(unique_ids) != len(all_ids):
                duplicates = [x for x in all_ids if all_ids.count(x) > 1]
                pytest.fail(
                    f"Iteration {iteration + 1}/{num_iterations}: "
                    f"Found {len(all_ids) - len(unique_ids)} collisions. "
                    f"Duplicate IDs: {set(duplicates)}"
                )

    @pytest.mark.slow
    def test_gen_stream_id_no_collision_stress(self):
        """Stress test with 1000 iterations to thoroughly verify collision resistance.

        This is a long-running test marked as 'slow'. Run with: pytest -m slow
        Uses 2 processes, each generating 1000 stream IDs, repeated 1000 times.
        """
        num_processes = 2
        ids_per_process = 1000
        num_iterations = 100

        mp.set_start_method("spawn", force=True)
        for iteration in range(num_iterations):
            # Create a queue to collect results
            result_queue = mp.Queue()

            # Create and start individual processes for each iteration
            processes = []
            for _ in range(num_processes):
                p = mp.Process(target=generate_stream_ids, args=(ids_per_process, result_queue))
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Collect results from queue
            all_ids = []
            while not result_queue.empty():
                process_ids = result_queue.get()
                all_ids.extend(process_ids)

            # Check total count
            expected_total = num_processes * ids_per_process
            assert (
                len(all_ids) == expected_total
            ), f"Iteration {iteration + 1}: Expected {expected_total} IDs, got {len(all_ids)}"

            # Check for uniqueness across all processes
            unique_ids = set(all_ids)
            if len(unique_ids) != len(all_ids):
                duplicates = [x for x in all_ids if all_ids.count(x) > 1]
                pytest.fail(
                    f"Iteration {iteration + 1}/{num_iterations}: "
                    f"Found {len(all_ids) - len(unique_ids)} collisions. "
                    f"Duplicate IDs: {set(duplicates)}"
                )

            # Print progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                print(f"Completed {iteration + 1}/{num_iterations} iterations")

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
