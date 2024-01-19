from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

BITWISE_AND: ReduceOperation
BITWISE_OR: ReduceOperation
BITWISE_XOR: ReduceOperation
DESCRIPTOR: _descriptor.FileDescriptor
DOUBLE: DataType
FLOAT: DataType
HALF: DataType
INT16: DataType
INT32: DataType
INT64: DataType
INT8: DataType
LONG_DOUBLE: DataType
MAX: ReduceOperation
MIN: ReduceOperation
SUM: ReduceOperation
UINT16: DataType
UINT32: DataType
UINT64: DataType
UINT8: DataType

class AllgatherReply(_message.Message):
    __slots__ = ["receive_buffer"]
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...

class AllgatherRequest(_message.Message):
    __slots__ = ["rank", "send_buffer", "sequence_number"]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    rank: int
    send_buffer: bytes
    sequence_number: int
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ...) -> None: ...

class AllgatherVReply(_message.Message):
    __slots__ = ["receive_buffer"]
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...

class AllgatherVRequest(_message.Message):
    __slots__ = ["rank", "send_buffer", "sequence_number"]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    rank: int
    send_buffer: bytes
    sequence_number: int
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ...) -> None: ...

class AllreduceReply(_message.Message):
    __slots__ = ["receive_buffer"]
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...

class AllreduceRequest(_message.Message):
    __slots__ = ["data_type", "rank", "reduce_operation", "send_buffer", "sequence_number"]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    REDUCE_OPERATION_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    data_type: DataType
    rank: int
    reduce_operation: ReduceOperation
    send_buffer: bytes
    sequence_number: int
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ..., data_type: _Optional[_Union[DataType, str]] = ..., reduce_operation: _Optional[_Union[ReduceOperation, str]] = ...) -> None: ...

class BroadcastReply(_message.Message):
    __slots__ = ["receive_buffer"]
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...

class BroadcastRequest(_message.Message):
    __slots__ = ["rank", "root", "send_buffer", "sequence_number"]
    RANK_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    rank: int
    root: int
    send_buffer: bytes
    sequence_number: int
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ..., root: _Optional[int] = ...) -> None: ...

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class ReduceOperation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
