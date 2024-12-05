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
from functools import partial
from multiprocessing import shared_memory

from nvflare.app_opt.xgboost.histogram_based_v2.aggr import Aggregator

from .util import (
    bytes_to_int,
    ciphertext_to_int,
    encode_encrypted_numbers_to_str,
    encrypt_number,
    get_exponent,
    int_to_bytes,
    int_to_ciphertext,
)

SUFFIX = b"\xff"
SHARED_MEM_NAME = "encrypted_gh"


class Adder:
    def __init__(self, max_workers=10):
        self.exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self.num_workers = max_workers

    def add(self, encrypted_numbers, features, sample_groups=None, encode_sum=True):
        """

        Args:
            encrypted_numbers: list of encrypted numbers (combined gh), one for each sample
            features: list of tuples of (feature_id, mask, num_bins), one for each feature.
                    size of mask = size of encrypted_numbers: there is a bin number for each sample
                    num_bins specifies the number of bins for the feature
            sample_groups: list of sample groups, each group is a tuple of (group_id, id_list)
                    group_id is the group id, id_list is a list of sample IDs for which the add will be applied to
            encode_sum: if true, encode the sum into a JSON string

        Returns: list of tuples of (feature_id, group_id, sum), sum is the result of adding encrypted values of
            samples in the group for the feature.

        """

        shared_gh = shared_memory.ShareableList(self._shared_list(encrypted_numbers), name=SHARED_MEM_NAME)
        items = []

        for f in features:
            fid, mask, num_bins = f
            if not sample_groups:
                items.append((encode_sum, fid, mask, num_bins, 0, None))
            else:
                for g in sample_groups:
                    gid, sample_id_list = g
                    items.append((encode_sum, fid, mask, num_bins, gid, sample_id_list))

        pubkey = encrypted_numbers[0].public_key
        chunk_size = int((len(items) - 1) / self.num_workers) + 1

        results = self.exe.map(partial(_do_add, shared_gh.shm.name, pubkey), items, chunksize=chunk_size)
        rl = []
        for r in results:
            rl.append(r)

        shared_gh.shm.close()
        shared_gh.shm.unlink()

        return rl

    def _shared_list(self, encrypted_numbers: list) -> list:
        result = []
        for ciphertext in encrypted_numbers:
            # Due to a Python bug, a non-zero suffix is needed
            # See https://github.com/python/cpython/issues/10693
            result.append(int_to_bytes(ciphertext_to_int(ciphertext)) + SUFFIX)
            result.append(int_to_bytes(get_exponent(ciphertext)) + SUFFIX)

        return result


def shared_list_accessor(pubkey, shared_gh, index):
    """
    shared_gh contains ciphertext and exponent in bytes so each
    encrypted number takes 2 slots

    Due to the ShareableList bug, a non-zero byte is appended to the bytes
    """
    n = bytes_to_int(shared_gh[index * 2][:-1])
    exp = bytes_to_int(shared_gh[index * 2 + 1][:-1])
    ciphertext = int_to_ciphertext(n, pubkey=pubkey)
    return encrypt_number(pubkey, ciphertext, exp)


def _do_add(shared_mem_name, pubkey, item):

    shared_gh = shared_memory.ShareableList(name=shared_mem_name)
    encode_sum, fid, mask, num_bins, gid, sample_id_list = item
    # bins = [0 for _ in range(num_bins)]
    aggr = Aggregator()

    bins = aggr.aggregate(
        gh_values=shared_gh,
        sample_bin_assignment=mask,
        num_bins=num_bins,
        sample_ids=sample_id_list,
        accessor=partial(shared_list_accessor, pubkey),
    )

    if encode_sum:
        sums = encode_encrypted_numbers_to_str(bins)
    else:
        sums = bins

    shared_gh.shm.close()

    return fid, gid, sums
