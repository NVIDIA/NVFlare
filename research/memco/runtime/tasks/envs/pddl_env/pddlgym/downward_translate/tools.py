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


def cartesian_product(sequences):
    # TODO: Rename this. It's not good that we have two functions
    # called "product" and "cartesian_product", of which "product"
    # computes cartesian products, while "cartesian_product" does not.

    # This isn't actually a proper cartesian product because we
    # concatenate lists, rather than forming sequences of atomic elements.
    # We could probably also use something like
    # map(itertools.chain, product(*sequences))
    # but that does not produce the same results
    if not sequences:
        yield []
    else:
        temp = list(cartesian_product(sequences[1:]))
        for item in sequences[0]:
            for sequence in temp:
                yield item + sequence


def get_peak_memory_in_kb():
    try:
        # This will only work on Linux systems.
        with open("/proc/self/status") as status_file:
            for line in status_file:
                parts = line.split()
                if parts[0] == "VmPeak:":
                    return int(parts[1])
    except OSError:
        pass
    raise Warning("warning: could not determine peak memory")
