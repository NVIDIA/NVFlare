.. _cyclic:

Cyclic Workflow
---------------
.. currentmodule:: nvflare.app_common.workflows.cyclic_ctl.CyclicController

The cyclic workflow was added in NVIDIA FLARE 2.0 and implemented with the Controller API. It allows for a different control
flow implemented through the :meth:`control_flow()<control_flow>` method based on ``relay_and_wait()`` instead of
``broadcast_and_wait()`` in the :ref:`scatter_and_gather_workflow`.

:class:`nvflare.app_common.workflows.cyclic_ctl.CyclicController`

Example with Cyclic Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See the `Hello Cyclic Example <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-cyclic>`_ for an example application with
the cyclic workflow.
