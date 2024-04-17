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

        num_values = len(numbers)
        if num_values <= self.max_workers:
            w_values = [numbers]
            workers_needed = 1
        else:
            workers_needed = self.max_workers
            w_values = [None for _ in range(self.max_workers)]
            n = int(num_values / self.max_workers)
            w_load = [n for _ in range(self.max_workers)]
            r = num_values % self.max_workers
            if r > 0:
                for i in range(r):
                    w_load[i] += 1

            start = 0
            for i in range(self.max_workers):
                end = start + w_load[i]
                w_values[i] = numbers[start:end]
                start = end

        total_count = 0
        for v in w_values:
            total_count += len(v)
        assert total_count == num_values

        items = []
        for i in range(workers_needed):
            items.append((self.pubkey, w_values[i]))
        return self._encrypt(items)

    def _encrypt(self, items):
        results = self.exe.map(_do_enc, items)
        rl = []
        for r in results:
            rl.extend(r)
        return rl


def _do_enc(item):
    pubkey, numbers = item
    return numbers
