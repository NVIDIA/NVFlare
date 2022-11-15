Flare Object Serializer (FOBS)
==============================


Overview
--------

FOBS is a drop-in replacement for Pickle for security purposes. It uses **MessagePack** to
serialize objects.

FOBS sacrifices convenience for security. With Pickle, most objects are supported
automatically using introspection. To serialize an object using FOBS, a **Decomposer**
must be registered for the class. A few decomposers for commonly used classes are
pre-registered with the module.

FOBS supports enum types by registering decomposers automatically for all classes that
are subclasses of :code:`Enum`.

FOBS throws :code:`TypeError` exception when it encounters an object with no decomposer
registered. For example,
::
    TypeError: can not serialize 'xxx' object

Usage
-----

FOBS defines following 4 functions, similar to Pickle,

* :code:`dumps(obj)`: Serializes obj and returns bytes
* :code:`dump(obj, stream)`: Serializes obj and writes the result to stream
* :code:`loads(data)`: Deserializes the data and returns an object
* :code:`load(stream)`: Reads data from stream and deserializes it into an object


Examples,
::

    from nvflare.fuel.utils import fobs

    data = fobs.dumps(dxo)
    new_dxo = fobs.loads(data)

    # Pickle/json compatible functions can be used also
    data = fobs.dumps(shareable)
    new_shareable = fobs.loads(data)

Decomposers
-----------

Decomposers are classes that inherit abstract base class :code:`fobs.Decomposer`. FOBS
uses decomposers to break an object into **serializable objects** before serializing it
using MessagePack.

Decomposers are very similar to serializers, except that they don't have to convert object
into bytes directly, they can just break the object into other objects that are serializable.

An object is serializable if its type is supported by MessagePack or a decomposer is
registered for its class.

FOBS recursively decomposes objects till all objects are of types supported by MessagePack.
Decomposing looping must be avoided, which causes stack overflow. Decomposers form a loop
when one class is decomposed into another class which is eventually decomposed into the
original class. For example, this scenario forms the simplest loop: X decomposes into Y
and Y decomposes back into X.

MessagePack supports following types natively,

* None
* bool
* int
* float
* str
* bytes
* bytearray
* memoryview
* list
* dict

Decomposers for following classes are included with `fobs` module and auto-registered,

* tuple
* set
* OrderedDict
* datetime
* Shareable
* FLContext
* DXO
* Client
* RunSnapshot
* Workspace
* Signal
* AnalyticsDataType
* argparse.Namespace
* Learnable
* _CtxPropReq
* _EventReq
* _EventStats
* numpy.float32
* numpy.float64
* numpy.int32
* numpy.int64
* numpy.ndarray

All classes defined in :code:`fobs/decomposers` folder are automatically registered.
Other decomposers must be registered manually like this,

::

    fobs.register(FooDecomposer)
    fobs.register(BarDecomposer())


:code:`fobs.register` takes either a class or an instance as the argument. Decomposer whose
constructor takes arguments must be registered as instance.

A decomposer can either serialize the class into bytes or decompose it into objects of
serializable types. In most cases, it only involves saving members as a list and reconstructing
the object from the list.

Here is an example of a simple decomposer. Even though :code:`datetime` is not supported
by MessagePack, a decomposer is included in `fobs` module so no need to further decompose it.

::

    from nvflare.fuel.utils import fobs


    class Simple:

        def __init__(self, num: int, name: str, timestamp: datetime):
            self.num = num
            self.name = name
            self.timestamp = timestamp


    class SimpleDecomposer(fobs.Decomposer):

        def supported_type(self) -> Type[Any]:
            return Simple

        def decompose(self, obj) -> Any:
            return [obj.num, obj.name, obj.timestamp]

        def recompose(self, data: Any) -> Simple:
            return Simple(data[0], data[1], data[2])


    fobs.register(SimpleDecomposer)
    data = fobs.dumps(Simple(1, 'foo', datetime.now()))
    obj = fobs.loads(data)
    assert obj.num == 1
    assert obj.name == 'foo'
    assert isinstance(obj.timestamp, datetime)


The same decomposer can be registered multiple times. Only first one takes effect, the others
are ignored with a warning message.

Note that fobs_initialize() may need to be called if decomposers are not registered.

Enum Types
----------

FOBS supports enum types by default. Decomposers for all classes derived from :code:`Enum` are
automatically registered using the generic decomposer for enum.

In rare case that an enum class is too complicated that the generic decomposer can't
handle it, a special decomposer can be written and registered. This will prevent FOBS from
auto-registering the generic decomposer for this enum type.

The auto-registering of enum decomposers can be disabled like this,

::

    fobs.auto_register_enum_types(False)


Custom Types
------------

To support custom types with FOBS, the decomposers for the types must be included
with the custom code and registered.

The decomposers must be registered in both server and client code before FOBS is used.
A good place for registration is the constructors for controllers and executors. It
can also be done in `START_RUN` event handler.

Custom object cannot be put in `shareable` directly,
it must be serialized using FOBS first. Assuming `custom_data` contains custom type,
this is how data can be stored in shareable,
::
    shareable[CUSTOM_DATA] = fobs.dumps(custom_data)
On the receiving end,
::
    custom_data = fobs.loads(shareable[CUSTOM_DATA])

This doesn't work
::
    shareable[CUSTOM_DATA] = custom_data
