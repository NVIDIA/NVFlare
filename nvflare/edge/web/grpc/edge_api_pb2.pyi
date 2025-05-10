from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Request(_message.Message):
    __slots__ = ("type", "method", "header", "payload", "cookie")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    COOKIE_FIELD_NUMBER: _ClassVar[int]
    type: str
    method: str
    header: bytes
    payload: bytes
    cookie: bytes
    def __init__(self, type: _Optional[str] = ..., method: _Optional[str] = ..., header: _Optional[bytes] = ..., payload: _Optional[bytes] = ..., cookie: _Optional[bytes] = ...) -> None: ...

class Reply(_message.Message):
    __slots__ = ("status", "payload", "cookie")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    COOKIE_FIELD_NUMBER: _ClassVar[int]
    status: str
    payload: bytes
    cookie: bytes
    def __init__(self, status: _Optional[str] = ..., payload: _Optional[bytes] = ..., cookie: _Optional[bytes] = ...) -> None: ...
