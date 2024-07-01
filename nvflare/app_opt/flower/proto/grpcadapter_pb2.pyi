from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class MessageContainer(_message.Message):
    __slots__ = ["grpc_message_content", "grpc_message_name", "metadata"]
    class MetadataEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    GRPC_MESSAGE_CONTENT_FIELD_NUMBER: _ClassVar[int]
    GRPC_MESSAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    grpc_message_content: bytes
    grpc_message_name: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, metadata: _Optional[_Mapping[str, str]] = ..., grpc_message_name: _Optional[str] = ..., grpc_message_content: _Optional[bytes] = ...) -> None: ...
