.. _cyclic:

Cyclic Workflow
---------------
.. currentmodule:: nvflare.app_common.workflows.cyclic_ctl.CyclicController

Cyclic workflow is new for NVIDIA FLARE 2.0 and implemented with the Controller API. It allows for a different control
flow implemented through the :meth:`control_flow()<control_flow>` method based on ``relay_and_wait()`` instead of
``broadcast_and_wait()`` in the :ref:`scatter_and_gather_workflow`.

:class:`nvflare.app_common.workflows.cyclic_ctl.CyclicController`
