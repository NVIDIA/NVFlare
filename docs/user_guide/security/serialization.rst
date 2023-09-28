.. _serialization:

Message Serialization
=====================
Prior to NVFLARE 2.1, messages between the FL server and clients were serialized with Python's
`pickle <https://docs.python.org/3/library/pickle.html>` facility. Many people
have pointed out the potential security risks due to the flexibility of Pickle.

NVFLARE now uses a more secure mechanism called FOBS (Flare OBject Serializer) for message serialization and
deserialization when exchanging data between the server and clients.

See `<https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/utils/fobs/README.rst>`_ for usage guidelines.
