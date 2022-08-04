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

FOBS throws :code:`TypeError` exception when it encounters an object with no decomposer
registered.

Usage
-----

FOBS defines following 4 functions, similar to Pickle,

* :code:`serialize(obj)`: Serializes obj and returns bytes
* :code:`serialize_stream(obj, stream)`: Serializes obj and writes the result to stream
* :code:`deserialize(data)`: Deserializes the data and returns an object
* :code:`deserialize_stream(stream)`: Reads data from stream and deserializes it into an object

Following aliases are defined to make the function more familiar to Pickle/JSON users,
::
    load = deserialize_stream
    loads = deserialize
    dump = serialize_stream
    dumps = serialize

Examples,
::

    from nvflare.fuel.utils import fobs

    data = fobs.dumps(dxo)
    new_dxo = fobs.loads(data)

    # Pickle/json compatible functions can be used also
    data = fobs.dumps(shareable)
    new_shareable = fobs.loads(data)

Decomposers
~~~~~~~~~~~

Decomposers are classes that inherit abstract base class :code:`fobs.Decomposer`. FOBS
uses decomposers to break an object into **serializable objects** before serializing it
using MessagePack.

Decomposers are very similar to serializers, except that they don't have to convert object
into bytes directly, they can just break the object into other objects that are serializable.

An object is serializable if its type is supported by MessagePack or a decomposer is
registered for its class. MessagePack supports following types natively,

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

FOBS recursively decomposes objects till all objects are of types supported by MessagePack.
Decomposing looping must be avoided, which causes stack overflow. Decomposers form a loop
when one class is decomposed into another class which is eventually decomposed into the
original class. For example, this scenario forms the simplest loop: X decomposes into Y
and Y decomposes back into X.

Decomposers for following classes are included with `fobs` module and auto-registered,

* datetime
* numpy.float32
* numpy.float64
* numpy.int32
* numpy.int64
* numpy.ndarray
* Learnable
* Shareable
* FLContext
* DXO

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

        @staticmethod
        def supported_type() -> Type[Any]:
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
