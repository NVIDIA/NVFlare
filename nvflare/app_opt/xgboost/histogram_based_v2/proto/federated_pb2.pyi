from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HALF: _ClassVar[DataType]
    FLOAT: _ClassVar[DataType]
    DOUBLE: _ClassVar[DataType]
    LONG_DOUBLE: _ClassVar[DataType]
    INT8: _ClassVar[DataType]
    INT16: _ClassVar[DataType]
    INT32: _ClassVar[DataType]
    INT64: _ClassVar[DataType]
    UINT8: _ClassVar[DataType]
    UINT16: _ClassVar[DataType]
    UINT32: _ClassVar[DataType]
    UINT64: _ClassVar[DataType]

class ReduceOperation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MAX: _ClassVar[ReduceOperation]
    MIN: _ClassVar[ReduceOperation]
    SUM: _ClassVar[ReduceOperation]
    BITWISE_AND: _ClassVar[ReduceOperation]
    BITWISE_OR: _ClassVar[ReduceOperation]
    BITWISE_XOR: _ClassVar[ReduceOperation]
HALF: DataType
FLOAT: DataType
DOUBLE: DataType
LONG_DOUBLE: DataType
INT8: DataType
INT16: DataType
INT32: DataType
INT64: DataType
UINT8: DataType
UINT16: DataType
UINT32: DataType
UINT64: DataType
MAX: ReduceOperation
MIN: ReduceOperation
SUM: ReduceOperation
BITWISE_AND: ReduceOperation
BITWISE_OR: ReduceOperation
BITWISE_XOR: ReduceOperation

class AllgatherRequest(_message.Message):
    __slots__ = ("sequence_number", "rank", "send_buffer")
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    sequence_number: int
    rank: int
    send_buffer: bytes
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ...) -> None: ...

class AllgatherReply(_message.Message):
    __slots__ = ("receive_buffer",)
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...

class AllgatherVRequest(_message.Message):
    __slots__ = ("sequence_number", "rank", "send_buffer")
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    sequence_number: int
    rank: int
    send_buffer: bytes
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ...) -> None: ...

class AllgatherVReply(_message.Message):
    __slots__ = ("receive_buffer",)
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...

class AllreduceRequest(_message.Message):
    __slots__ = ("sequence_number", "rank", "send_buffer", "data_type", "reduce_operation")
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    REDUCE_OPERATION_FIELD_NUMBER: _ClassVar[int]
    sequence_number: int
    rank: int
    send_buffer: bytes
    data_type: DataType
    reduce_operation: ReduceOperation
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ..., data_type: _Optional[_Union[DataType, str]] = ..., reduce_operation: _Optional[_Union[ReduceOperation, str]] = ...) -> None: ...

class AllreduceReply(_message.Message):
    __slots__ = ("receive_buffer",)
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...

class BroadcastRequest(_message.Message):
    __slots__ = ("sequence_number", "rank", "send_buffer", "root")
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    SEND_BUFFER_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    sequence_number: int
    rank: int
    send_buffer: bytes
    root: int
    def __init__(self, sequence_number: _Optional[int] = ..., rank: _Optional[int] = ..., send_buffer: _Optional[bytes] = ..., root: _Optional[int] = ...) -> None: ...

class BroadcastReply(_message.Message):
    __slots__ = ("receive_buffer",)
    RECEIVE_BUFFER_FIELD_NUMBER: _ClassVar[int]
    receive_buffer: bytes
    def __init__(self, receive_buffer: _Optional[bytes] = ...) -> None: ...
