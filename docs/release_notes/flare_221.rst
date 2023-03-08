What's New in FLARE v2.2.1
==========================

Using FOBS to serialize/deserialize data between Client and Server
------------------------------------------------------------------
Prior to NVFLARE 2.1.4, NVFLARE used python's `pickle <https://docs.python.org/3/library/pickle.html>`_ to transfer data between the FL clients and server.
NVFLARE now uses the FLARE Object Serializer (FOBS). You might experience failures if your code is still using Pickle. 
To migrate the code or if you experience errors due to this, please refer to `Flare Object Serializer (FOBS) <https://github.com/NVIDIA/NVFlare/blob/dev/nvflare/fuel/utils/fobs/README.rst>`_.

Another type of failure is due to data types that are not supported by FOBS. By default FOBS supports some data types, if the data type (Custom Class or Class from 3rd parties)
is not part of supported FOBS data type, then you need to follow the instructions at
`Flare Object Serializer (FOBS) <https://github.com/NVIDIA/NVFlare/blob/dev/nvflare/fuel/utils/fobs/README.rst>`_.

Essentially, to address this type of issue, you need to do the following steps:
  - Create a FobDecomposer class for the targeted data type
  - Register the newly created FobDecomposer before the data type is transmitted between client and server.

The following examples are directly copied from `Flare Object Serializer (FOBS) <https://github.com/NVIDIA/NVFlare/blob/dev/nvflare/fuel/utils/fobs/README.rst>`_.

.. code-block:: python

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

Register the data type in FOBS before the data type is used, then you can register the newly created FOBDecomposer

.. code-block:: python

    fobs.register(SimpleDecomposer)

.. note::

  The decomposers must be registered in both server and client code before FOBS is used.
  A good place for registration is the constructors for the controllers and executors. It can also be done in the START_RUN event handler.

Use FOBS to serialize data before you use sharable
""""""""""""""""""""""""""""""""""""""""""""""""""
A custom object cannot be put in shareable directly, it must be serialized using FOBS first.
Assuming custom_data contains custom type, this is how data can be stored in shareable:

.. code-block:: python

    shareable[CUSTOM_DATA] = fobs.dumps(custom_data)

On the receiving end:

.. code-block:: python

    custom_data = fobs.loads(shareable[CUSTOM_DATA])


.. note::

  This does not work:

  .. code-block:: python
  
    shareable[CUSTOM_DATA] = custom_data


New local directory
-------------------
With 2.2.1, the provision command will produce not only the ``startup`` directory, but a ``local`` directory. 
The resource allocation that used to be in ``project.yml`` is now expected in a ``resources.json`` file in this new ``local`` directory, and each
sites/clients needs to manage this separately for each location.
You need to place/modify your own site's ``authorization.json`` and ``privacy.json`` files in the ``local`` directory as well if you want to
change the default policies. 

The default configurations are provided in each site's local directory:

.. code-block::

    local
    ├── authorization.json.default
    ├── log.config.default
    ├── privacy.json.sample
    └── resources.json.default

These defaults can be overridden by removing the default suffix and modifying the configuration as needed for the specific site.
