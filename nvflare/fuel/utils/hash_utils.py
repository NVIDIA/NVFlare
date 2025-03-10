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
MAX_NUM_BUCKETS = 64 * 1024


class UniformHash:
    """A hash algorithm with uniform distribution. It achieves this with following steps,
    1. Get a hash value of the key using first 8 bytes of SHA256
    2. Map the hash value to a virtual hash table using modulo operation
    3. Map the virtual bucket to real bucket by even distribution

    The virtual hash is needed because real hash table size may not be a prime number and
    there are known issues of modulo operation.

    """

    def __init__(self, num_buckets: int):
        """
        Initialize the hash function
        Args:
            num_buckets: Number of buckets, e.g. Number of servers to distribute the load
        """
        check_number_range("num_buckets", num_buckets, 1, MAX_NUM_BUCKETS)
        self.num_buckets = num_buckets
        self.virtual_hashes_per_bucket = PRIME // num_buckets

    def get_num_buckets(self) -> int:
        """
        Get the number of buckets
        Returns:
             Number of buckets
        """

        return self.num_buckets

    def hash(self, key: str) -> int:
        """
        Hash the key to a bucket index
        Args:
            key: A string key to be hashed
        Returns:
            The bucket index between 0 and num_buckets-1
        """
        check_str("key", key)

        # Step 1, calculate hash value using first 8 bytes of SHA256
        sha_bytes = hashlib.sha256(key.encode()).digest()
        sha = int.from_bytes(sha_bytes[:8], "big")

        # Step 2, map the hash value to a virtual hash table whose size is PRIME using modulo operation
        virtual_hash = sha % PRIME

        # Step 3, evenly distribute the virtual hash to real buckets
        # n is a float number representing the bucket index
        n = virtual_hash / self.virtual_hashes_per_bucket
        if n < self.num_buckets:
            index = int(n)
        else:
            # The last bucket may have more virtual hashes than others
            # Evenly spread the extra to the first few buckets
            index = virtual_hash % self.num_buckets

        return index
