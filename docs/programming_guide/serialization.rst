.. _serialization:

Serialization
=============

Due to security concerns, `pickle <https://docs.python.org/3/library/pickle.html>` has been replaced with FOBS (Flare object serialization) in NVFlare to exchange data between the server and clients.
See `<https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/utils/fobs/README.rst>`_ for usage guidelines.
