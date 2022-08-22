.. _serialization:

Serialization
=============

Due to security concerns, `pickle <https://docs.python.org/3/library/pickle.html>` should not be used for serialization. FOBS (Flare object serialization) is a replacement included in NVFlare and
should be used instead to exchange data between the server and clients. See `<https://github.com/NVIDIA/NVFlare/blob/2.1/nvflare/fuel/utils/fobs/README.rst>`_ for usage guidelines.
