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
