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

"""Model exchange protocol specification and utilities.

This module defines the complete protocol for exchanging models between components,
including message format specification, supported types, and validation utilities.

Example:
    >>> from nvflare.edge.utils.model_protocol import ModelBufferType, ModelExchangeFormat
    >>> payload = {
    ...     ModelExchangeFormat.MODEL_BUFFER: encoded_data,
    ...     ModelExchangeFormat.MODEL_BUFFER_TYPE: ModelBufferType.EXECUTORCH,
    ...     ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.BINARY,
    ...     ModelExchangeFormat.MODEL_BUFFER_ENCODING: ModelEncoding.BASE64
    ... }
    >>> verified_payload = verify_payload(
    ...     payload,
    ...     expected_type=ModelBufferType.EXECUTORCH
    ... )
"""

import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)


class ModelBufferType:
    """Supported model buffer types for data exchange.

    These constants define the supported types of data that can be exchanged,
    helping ensure consistency across different components.
    """

    # Model formats
    EXECUTORCH = "executorch"


class ModelNativeFormat:
    """Native format of the data before any transmission encoding."""

    BINARY = "binary"  # Raw bytes, binary data
    STRING = "string"  # Text-based data


class ModelEncoding:
    """Supported encodings for data transmission.

    For binary native format:
        - BASE64 or HEX encoding required for safe transmission

    For string native format:
        - UTF8 or ASCII for character encoding
        - NONE for plain string data
    """

    BASE64 = "base64"
    HEX = "hex"
    UTF8 = "utf8"
    ASCII = "ascii"
    NONE = None


class ModelExchangeFormat:
    """Constants for model exchange protocol between components.

    The protocol uses three main attributes:
    1. TYPE: What the data represents (e.g., executorch, pytorch)
    2. NATIVE_FORMAT: Original format before any transmission encoding
    3. ENCODING: How the data is encoded for transmission

    Format and Encoding combinations:
        - For binary FORMAT:
            - base64 encoding (safe for text transmission)
            - hex encoding (alternative text encoding)
        - For string FORMAT:
            - utf8 encoding (for unicode text)
            - ascii encoding (for restricted character set)
            - null/empty (for plain string)

    Examples:
        Binary data (ExecutorTorch model):
            {
                MODEL_BUFFER: <encoded_data>,
                MODEL_BUFFER_TYPE: ModelBufferType.EXECUTORCH,
                MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.BINARY,
                MODEL_BUFFER_ENCODING: ModelEncoding.BASE64
            }

        String data (JSON config):
            {
                MODEL_BUFFER: <json_string>,
                MODEL_BUFFER_TYPE: ModelBufferType.JSON,
                MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.STRING,
                MODEL_BUFFER_ENCODING: ModelEncoding.UTF8
            }
    """

    MODEL_BUFFER = "model_buffer"
    MODEL_BUFFER_SIZE = "model_buffer_size"  # Optional
    MODEL_BUFFER_TYPE = "model_buffer_type"
    MODEL_BUFFER_NATIVE_FORMAT = "model_buffer_native_format"
    MODEL_BUFFER_ENCODING = "model_buffer_encoding"


def verify_payload(
    task_data: Dict,
    expected_type: Optional[str] = None,
    expected_format: Optional[str] = None,
    expected_encoding: Optional[str] = None,
) -> Dict:
    """Verify that the task data payload follows the model exchange protocol.

    Args:
        task_data: The task data dictionary to verify
        expected_type: Expected model buffer type (from ModelBufferType)
        expected_format: Expected native format (from ModelNativeFormat)
        expected_encoding: Expected encoding (from ModelEncoding)

    Returns:
        Dict: The validated payload dictionary

    Raises:
        ValueError: If the payload structure is invalid or values don't match expected
    """
    if not isinstance(task_data, dict):
        raise ValueError("Task data must be a dictionary")

    # Validate required fields
    required_fields = [
        ModelExchangeFormat.MODEL_BUFFER,
        ModelExchangeFormat.MODEL_BUFFER_TYPE,
        ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT,
        ModelExchangeFormat.MODEL_BUFFER_ENCODING,
    ]

    for field in required_fields:
        if field not in task_data:
            raise ValueError(f"Missing required field: {field}")

    # Validate expected values if provided
    if expected_type and task_data[ModelExchangeFormat.MODEL_BUFFER_TYPE] != expected_type:
        raise ValueError(
            f"Expected model type {expected_type}, " f"got {task_data[ModelExchangeFormat.MODEL_BUFFER_TYPE]}"
        )

    if expected_format and task_data[ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT] != expected_format:
        raise ValueError(
            f"Expected native format {expected_format}, "
            f"got {task_data[ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT]}"
        )

    if expected_encoding and task_data[ModelExchangeFormat.MODEL_BUFFER_ENCODING] != expected_encoding:
        raise ValueError(
            f"Expected encoding {expected_encoding}, " f"got {task_data[ModelExchangeFormat.MODEL_BUFFER_ENCODING]}"
        )

    return task_data
