# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import hashlib

from nvflare.fuel.utils.validation_utils import check_number_range, check_str

# A large prime number as virtual hash table size
PRIME = 100003
MAX_NUM_BUCKETS = 64*1024


class UniformHash:
    """A hash algorithm with uniform distribution. It achieves this with following steps,
    1. Get a hash value using SHA256
    2. Map the hash value to a virtual hash table
    3. Map the virtual bucket to real bucket

    """

    def __init__(self, num_buckets: int):
        check_number_range("num_buckets", num_buckets, 1, MAX_NUM_BUCKETS)
        self.num_buckets = num_buckets
        self.virtual_hashes_per_bucket = PRIME // num_buckets

    def get_num_buckets(self) -> int:
        return self.num_buckets

    def hash(self, key: str) -> int:
        check_str("key", key)
        # The hash() function changes value every run so SHA256 is used
        sha_bytes = hashlib.sha256(key.encode()).digest()
        sha = int.from_bytes(sha_bytes[:8], "big")
        virtual_hash = sha % PRIME

        index = virtual_hash // self.virtual_hashes_per_bucket
        if index >= self.num_buckets:
            index = virtual_hash % self.num_buckets

        return index
