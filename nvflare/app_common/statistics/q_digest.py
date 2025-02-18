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


class QDigest:
    def __init__(self, bin_precision=1000):
        """
        :param bin_precision: Controls binning for floating-point numbers (higher = more precise)
        """
        # Start with an undefined range
        self.min_val = 0
        self.max_val = 0
        self.bin_precision = bin_precision  # Defines granularity for floats
        self.tree = {}  # Stores frequency counts
        self.total_count = 0

    def _get_node_index(self, value):
        """Maps float or int to a discrete bin index."""
        scaled_value = round((value - self.min_val) * self.bin_precision)
        # Use self.max_val to constrain the index, avoiding recursion
        return max(0, min(scaled_value, int((self.max_val - self.min_val) * self.bin_precision)))

    def insert(self, value):

        # Dynamically update min and max values
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

        """Insert a value into the Q-Digest."""
        node = self._get_node_index(value)
        self.tree[node] = self.tree.get(node, 0) + 1
        self.total_count += 1

    def quantile(self, q):

        if not self.tree.keys():
            raise ValueError("empty tree in QDigest")

        """Approximate the q-th quantile."""
        threshold = self.total_count * q
        cumulative = 0
        sorted_keys = sorted(self.tree.keys())

        for key in sorted_keys:
            cumulative += self.tree[key]
            if cumulative >= threshold:
                return (key / self.bin_precision) + self.min_val  # Reverse scaling

        # If no quantile threshold is met, return the highest available key value.
        print("Quantile not found within available data range, return the highest available value")

        print("sorted_keys=", sorted_keys)

        last_key = sorted_keys[-1]
        return (last_key / self.bin_precision) + self.min_val

    def merge(self, other):
        """Merge another Q-Digest into this one."""
        # Adjust min_val and max_val dynamically based on the other Q-Digest
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)

        # Dynamically adjust the bin_precision (take the maximum for accuracy)
        self.bin_precision = max(self.bin_precision, other.bin_precision)

        for key, count in other.tree.items():
            self.tree[key] = self.tree.get(key, 0) + count
        self.total_count += other.total_count

    # Serialize the Q-Digest into a dictionary
    def serialize(self):
        return {
            "min_val": self.min_val,
            "max_val": self.max_val,
            "bin_precision": self.bin_precision,
            "tree": self.tree,
            "total_count": self.total_count,
        }

    # Deserialize the dictionary to recreate a Q-Digest object
    @classmethod
    def deserialize(cls, data):
        q_digest = cls(bin_precision=data["bin_precision"])
        q_digest.min_val = data["min_val"]
        q_digest.max_val = data["max_val"]
        q_digest.tree = data["tree"]
        q_digest.total_count = data["total_count"]
        return q_digest
