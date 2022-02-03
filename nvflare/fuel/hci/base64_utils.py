# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import base64


def bytes_to_b64str(data_bytes) -> str:
    """Convert binary to base64-encoded string."""
    encoded_bytes = base64.b64encode(data_bytes)
    return encoded_bytes.decode("ascii")


def b64str_to_bytes(b64str: str) -> bytes:
    """Convert base64-encoded string to binary."""
    encoded_bytes = b64str.encode("ascii")
    return base64.b64decode(encoded_bytes)


def binary_file_to_b64str(file_name) -> str:
    """Encode content of a binary file to a Base64-encoded ASCII string.

    Args:
        file_name: the binary file to be processed

    Returns: base64-encoded ASCII string

    """
    data_bytes = open(file_name, "rb").read()
    return bytes_to_b64str(data_bytes)


def b64str_to_binary_file(b64str: str, file_name):
    """Decode a base64-encoded string and write it into a binary file.

    Args:
        b64str: the base64-encoded ASCII string
        file_name: the file to write to

    Returns: number of bytes written

    """
    data_bytes = b64str_to_bytes(b64str)
    with open(file_name, "wb") as f:
        f.write(data_bytes)
    return len(data_bytes)


def text_file_to_b64str(file_name) -> str:
    """Encode content of a text file to a Base64-encoded ASCII string.

    Args:
        file_name: name of the text file

    Returns: base64-encoded string

    """
    data_string = open(file_name, "r").read()
    data_bytes = data_string.encode("utf-8")
    return bytes_to_b64str(data_bytes)


def b64str_to_text_file(b64str: str, file_name):
    """Decode a base64-encoded string and write result into a text file.

    Args:
        b64str: base64-encoded string
        file_name: file to be created

    Returns: number of data types written (may not be the same as number of characters)

    """
    data_bytes = b64str_to_bytes(b64str)
    data_string = data_bytes.decode("utf-8")
    with open(file_name, "w") as f:
        f.write(data_string)
    return len(data_bytes)
