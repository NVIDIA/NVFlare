.. _fl_component:

FLComponent
===========
.. currentmodule:: nvflare.apis.fl_component.FLComponent

:class:`nvflare.apis.fl_component.FLComponent` is the base class of all the FL components. Executors, controllers, filters, aggregators, and their subtypes for
example trainer are all FLComponents now.

.. literalinclude:: ../../nvflare/apis/fl_component.py
    :language: python
    :lines: 7-21

Each ``FLComponent`` is automatically added as an event handler in the system when a new instance is created.
You can implement the :meth:`handle_event<handle_event>` to plugin additional customized actions to the FL workflows.

To fire events, :meth:`fire_event<fire_event>` can be used, and :meth:`fire_fed_event<fire_fed_event>` can be used to
fire an event across participants.

The logging methods :meth:`log_debug<log_debug>`, :meth:`log_info<log_info>`, :meth:`log_warning<log_warning>`,
:meth:`log_error<log_error>`, and :meth:`log_exception<log_exception>` should be used to prefix log messages with
contextual information and integrate with other system features.

In extreme cases where the system encounters errors that prevent further operation, :meth:`task_panic<task_panic>` can
be called to end the task, or :meth:`system_panic<system_panic>` can be called to end the run.

Default data in the built-in FLComponents
-----------------------------------------
For the built-in FLComponents provided by NVIDIA FLARE, we assure the following data is set in the ``Shareable`` and ``FLContext``.

You can also define the structure of ``Sharable`` objects that fits your needs and
add your training associated data into ``FLContext``.
