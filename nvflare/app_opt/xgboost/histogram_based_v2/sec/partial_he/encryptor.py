# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import concurrent.futures


class Encryptor:
    def __init__(self, pubkey, max_workers=10):
        self.max_workers = max_workers
        self.pubkey = pubkey
        self.exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    def encrypt(self, numbers):
        """
        Encrypt a list of clear text numbers
        Args:
            numbers: clear text numbers to be encrypted
        Returns: list of encrypted numbers
        """
        items = [(self.pubkey, numbers[i]) for i in range(len(numbers))]
        chunk_size = int(len(items) / self.max_workers)
        if chunk_size == 0:
            chunk_size = 1

        results = self.exe.map(_do_enc, items, chunksize=chunk_size)
        rl = []
        for r in results:
            rl.append(r)
        return rl


def _do_enc(item):
    pubkey, num = item
    return pubkey.encrypt(num)
