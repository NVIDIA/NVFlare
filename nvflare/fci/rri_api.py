#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from io import BytesIO
from typing import Optional, Any

from nvflare.fci.endpoint import Endpoint
from nvflare.fci.headers import Headers
from nvflare.fci.responders import BytesResponder, ObjectResponder, StreamResponder


def request_bytes(endpoint: Endpoint, channel: int,
                  headers: Headers, payload: bytes, timeout_ms=0) -> (Headers, bytes):
    """Send request to endpoint/channel and wait for response

    This method is similar to a HTTP or RPC call.

    Args:
        endpoint: An endpoint to send the request to
        channel: Channel number
        headers: Headers, optional
        payload: The binary payload
        timeout_ms: Timeout in milliseconds, 0 means system default

    Returns:
        A tuple with response header and response data as bytes

    Raises:
        CommError: If any error happens while sending the request
    """

    pass


def register_bytes_responder(endpoint: Optional[Endpoint], channel: int, headers, responder: BytesResponder):
    """Register a responder to respond to FCI request

    This is similar to HTTP server handler or gRPC servicer method

    Args:
        endpoint: Endpoint of the request, None to handle requests from all endpoints
        channel: Channel number
        headers: Headers, optional
        responder: The class to handle the request

    Raises:
        CommError: If duplicate endpoint/channel or responder is of wrong type
    """

    pass


def request_object(endpoint: Endpoint, channel: int, headers: Headers, data: Any, timeout_ms=0) -> (Headers, Any):
    """Send request to endpoint/channel and wait for response as Python object

    This method is similar to HTTP or RPC call.

    Args:
        endpoint: An endpoint to send the request to
        channel: Channel number
        headers: Headers, optional
        data: A Python object as request
        timeout_ms: Timeout in milliseconds, 0 means system default

    Returns:
        A tuple with response header and response data as a Python object

    Raises:
        CommError: If any error happens while sending the request
    """
    pass


def register_object_responder(endpoint: Optional[Endpoint], channel: int,
                              headers: Headers, responder: ObjectResponder):
    """Register a responder to respond to FCI request as Python object

    This is similar to HTTP server handler or gRPC servicer method

    Args:
        endpoint: Endpoint of the request, None to handle requests from all endpoints
        channel: Channel number
        headers: Headers, optional
        responder: The class to handle the request as object

    Raises:
        CommError: If duplicate endpoint/channel or responder is of wrong type
    """

    pass


def request_stream(endpoint: Endpoint, channel: int,
                   headers: Headers, out_stream: BytesIO, timeout_ms=0) -> (Headers, BytesIO):
    """Send a streaming request to endpoint/channel and wait for response as a stream

    This method is similar to a streamed HTTP or RPC call.

    Args:
        endpoint: An endpoint to send the request to
        channel: Channel number
        headers: Headers, optional
        out_stream: A stream to write request to, close stream to finish the request
        timeout_ms: Timeout in milliseconds, 0 means system default

    Returns:
        A tuple with response header and response data as an input stream, close stream to indicate processing is done

    Raises:
        CommError: If any error happens while sending the request
    """

    pass


def register_stream_responder(endpoint: Optional[Endpoint], channel: int, headers, responder: StreamResponder):
    """Register a responder to respond to FCI request as stream

    This is similar to streamed version of HTTP server handler or gRPC servicer method

    Args:
        endpoint: Endpoint of the request, None to handle requests from all endpoints
        channel: Channel number
        headers: Headers, optional
        responder: The class to handle the request as stream, close the stream to finish the response.

    Raises:
        CommError: If duplicate endpoint/channel or responder is of wrong type
    """

    pass
