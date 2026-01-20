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

"""
Standalone TensorBoard event file reader that doesn't require TensorFlow.
This avoids C++ mutex locking issues by reading the protobuf files directly in pure Python.
"""

import struct
import sys

# Try to import tensorflow protobuf definitions
try:
    from tensorflow.core.util import event_pb2

    HAS_TF_PROTO = True
except ImportError:
    HAS_TF_PROTO = False
    # Fallback: try tensorboard standalone
    try:
        from tensorboard.compat.proto import event_pb2

        HAS_TF_PROTO = True
    except ImportError:
        pass

if not HAS_TF_PROTO:
    print("ERROR: Cannot import protobuf definitions.")
    print("Please ensure either 'tensorflow' or 'tensorboard' is installed.")
    sys.exit(1)


def read_tfevents_file(filepath, tags=None):
    """
    Read a TensorBoard event file and extract scalar values.

    Args:
        filepath: Path to the tfevents file
        tags: List of tag names to extract (None = all tags)

    Returns:
        Dictionary mapping tag names to list of [step, value] pairs
    """
    data = {}

    try:
        with open(filepath, "rb") as f:
            while True:
                # TensorBoard record format:
                # uint64    length
                # uint32    masked_crc32_of_length
                # byte      data[length]
                # uint32    masked_crc32_of_data

                # Read header: uint64 length (8 bytes, little-endian)
                header_bytes = f.read(8)
                if len(header_bytes) < 8:
                    break  # End of file

                # Unpack length (little-endian uint64)
                data_len = struct.unpack("<Q", header_bytes)[0]

                # Read and skip the masked CRC of length (4 bytes)
                f.read(4)

                # Read the actual event data
                event_bytes = f.read(data_len)
                if len(event_bytes) < data_len:
                    break  # Incomplete event at end of file

                # Read and skip footer CRC (4 bytes)
                f.read(4)

                # Parse the event protobuf
                event = event_pb2.Event()
                event.ParseFromString(event_bytes)

                # Extract scalar summaries
                if event.HasField("summary"):
                    for value in event.summary.value:
                        # Check if this is a scalar value
                        if value.HasField("simple_value"):
                            # Filter by tags if specified
                            if tags is None or value.tag in tags:
                                if value.tag not in data:
                                    data[value.tag] = []
                                data[value.tag].append([event.step, value.simple_value])

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        import traceback

        traceback.print_exc()

    return data


def get_available_tags(filepath):
    """Get all available scalar tags in a tfevents file."""
    tags = set()

    try:
        with open(filepath, "rb") as f:
            while True:
                header_bytes = f.read(8)
                if len(header_bytes) < 8:
                    break

                data_len = struct.unpack("<Q", header_bytes)[0]
                f.read(4)  # skip CRC

                event_bytes = f.read(data_len)
                if len(event_bytes) < data_len:
                    break

                f.read(4)  # skip CRC

                event = event_pb2.Event()
                event.ParseFromString(event_bytes)

                if event.HasField("summary"):
                    for value in event.summary.value:
                        if value.HasField("simple_value"):
                            tags.add(value.tag)

    except Exception as e:
        print(f"Error scanning {filepath}: {e}")

    return sorted(list(tags))
