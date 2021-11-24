.. _data_exchange_object:

Data Exchange Object (DXO)
==========================
.. currentmodule:: nvflare.apis.dxo.DXO

The Data Exchange Format (:class:`nvflare.apis.dxo.DXO`) in NVIDIA FLARE standardizes the data passed between the communicating parties.

.. literalinclude:: ../../nvflare/apis/dxo.py
    :language: python
    :lines: 29-40

``data_kind`` keeps track of the kind of data for example "WEIGHTS" or "WEIGHT_DIFF".

``meta`` is a dict that can contain additional properties.

The method :meth:`to_shareable()<to_shareable>` produces a :ref:`shareable`, and a DXO can be retrieved from a
:ref:`shareable` with :meth:`nvflare.apis.dxo.from_shareable`.

It is recommended to use DXO to maintain consistency in managing the data throughout the FL system.
