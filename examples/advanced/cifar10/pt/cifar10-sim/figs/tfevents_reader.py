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
Standalone TensorBoard event file reader using tbparse.
This avoids TensorFlow dependencies and provides a simple interface.
"""

import os
import sys

try:
    from tbparse import SummaryReader
except ImportError:
    print("ERROR: Cannot import tbparse.")
    print("Please install it: pip install tbparse")
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
        # tbparse.SummaryReader expects a directory, not a file
        # So we pass the directory containing the event file
        log_dir = os.path.dirname(filepath) if os.path.isfile(filepath) else filepath

        # Use tbparse to read the event file(s) in the directory
        reader = SummaryReader(log_dir, pivot=False)
        df = reader.scalars

        if df.empty:
            return data

        # Filter by tags if specified
        if tags is not None:
            df = df[df["tag"].isin(tags)]

        # Convert to the expected format: {tag: [[step, value], ...]}
        for tag in df["tag"].unique():
            tag_df = df[df["tag"] == tag][["step", "value"]].sort_values("step")
            data[tag] = tag_df.values.tolist()

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        import traceback

        traceback.print_exc()

    return data


def get_available_tags(filepath):
    """Get all available scalar tags in a tfevents file."""
    tags = set()

    try:
        # tbparse.SummaryReader expects a directory, not a file
        log_dir = os.path.dirname(filepath) if os.path.isfile(filepath) else filepath

        reader = SummaryReader(log_dir, pivot=False)
        df = reader.scalars

        if not df.empty and "tag" in df.columns:
            tags = set(df["tag"].unique())

    except Exception as e:
        print(f"Error scanning {filepath}: {e}")

    return sorted(list(tags))
