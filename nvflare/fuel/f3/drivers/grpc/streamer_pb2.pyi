from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Frame(_message.Message):
    __slots__ = ["data", "seq"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    seq: int
    def __init__(self, seq: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...
