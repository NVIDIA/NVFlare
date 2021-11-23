NVIDIA FLARE Event Mechanism
============================
NVIDIA FLARE comes with a powerful event mechanism that allows dynamic notifications sent to all objects that are of a
subclass of :class:`FLComponent<nvflare.apis.fl_component.FLComponent>`.

All component types in NVIDIA FLARE (e.g. Filter, Executor, Responder, Controller, Widget, etc.) are subclasses of
``FLComponent``, hence they are all event handlers. You can create additional subclasses, and their objects will
automatically become event handlers too.

Events
------
Events represent some important moments during the execution of the system logic. Following are typical examples of
such moments:

    - Before some action happens (e.g. before aggregation)
    - After some action happens (e.g. after aggregation)
    - When some important data becomes available (e.g. best model updated)

You can freely invent your own event types and fire events in your processing logic.

Event Type
^^^^^^^^^^
An event type is simply a string. Since anyone can invent a new event type, event naming conventions should be defined
to avoid possible name conflicts.

Event Data
^^^^^^^^^^
Event data can be anything. Most (but not all) event types have data. Event data is placed into the FLContext object as
a property under a well-defined name. Event handlers that listen to the event type get the event data with FLContext's
get_prop() method.

Not all event types have data - some events are merely used for the timing of the events.

Firing Events
^^^^^^^^^^^^^
To fire an event in your processing logic:
    - First, decide when to fire the event in your processing logic
    - Second, if the event has some data, add it to the fl_ctx. Note that in most cases you should already have a
      fl_ctx. If not, you can create a new one with engine.new_context().
    - Finally, fire the event.

The following is a typical code pattern::

    fl_ctx.set_prop(key="yourEventDataKey",
                      data=yourEventData,
                      private=True,
                      sticky=False)
    engine = fl_ctx.get_engine()
    engine.fire_event(event_type="youEventTypeName",
                        fl_ctx=fl_ctx)

.. note::

    Usually the event data should be private and non-sticky. But if you also want to keep the data around permanently
    during the RUN, you can set it to sticky.

.. note::

    If your event could have multiple pieces of data, you can simply call fl_ctx.set_prop() multiple times - one for
    each piece of data.

If your event doesn't have any data, then you can skip the first step.

Handling Events
^^^^^^^^^^^^^^^
When an event is fired, the handle_event() method of all event handling components are invoked to process the event.

.. note::

    The order in which the components are invoked is non-deterministic. Therefore you cannot rely on a component being
    invoked before another. This behavior may change in the future if needed.

The following is a typical code pattern for handling events you are interested in::

    def handle_event(self, event_type: str, fl_ctx: FLContext):
       if event_type == 'event1':
           event_data = fl_ctx.get_prop('eventDataKeyName')
           ...
       elif event_type == 'event2':
           # process

Event handlers are called in sequence. The failure (exception) of one handler does not stop the invocation of subsequent handlers.

Built-in Event Types
--------------------
NVIDIA FLARE's system-defined event types are specified in :class:`nvflare.apis.event_type.EventType`:

todo: go over these

.. csv-table::
   :header: Event, Description

    START_RUN, Start of the current "run"
    END_RUN, End of the current "run"
    CLIENT_REGISTER, When the client register to the server
    CLIENT_QUIT, When the client quit from the server.
    GET_PULL_REQUEST, .
    GET_SUBMIT_RESULT, .
    BEFORE_PULL_TASK, Before the .
    AFTER_PULL_TASK, After the .
    COLLECT_STATUS, .
    BEFORE_PROCESS_SUBMISSION, Before the .
    AFTER_PROCESS_SUBMISSION, After the .
    BEFORE_TASK_DATA_FILTER, Before the .
    AFTER_TASK_DATA_FILTER, After the .
    BEFORE_TASK_RESULT_FILTER, Before the .
    AFTER_TASK_RESULT_FILTER, After the .
    BEFORE_TASK_EXECUTION, Before the .
    AFTER_TASK_EXECUTION, After the .
    BEFORE_SEND_TASK_RESULT, Before the .
    AFTER_SEND_TASK_RESULT, After the .

RUN Lifecycle Events
--------------------
The most important event types of all are START_RUN and END_RUN.

In NVIDIA FLARE, a FL experiment is conducted in a RUN. During the course of a FL study, researchers usually need to conduct
many RUNs to achieve expected results.

START_RUN event occurs when a new RUN is about to start, usually triggered by the researcher via admin commands. If your
component needs to be initialized, you must listen to this event type and get your component ready for work.

The END_RUN event occurs when the RUN is about to end, usually triggered by completion of the workflow or the abort
command from the researcher. If needed, you can use this event to gracefully finalize and/or clean up your component.

If your component provides some services that other components can use, you can put your component as a private and
sticky prop into the fl_ctx at the time of START_RUN under a uniquely defined prop name. Other components can later get
your component with this name and invoke the services of your component.
