.. _event_system:

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

.. csv-table::
   :header: Event, Description, Data Key, Data Type, Server, Client

    START_RUN,A new RUN is about to start,fl_ctx.get_run_number(),int,X,X
    END_RUN,The current RUN is about to end,fl_ctx.get_run_number(),int,X,X
    START_WORKFLOW,Workflow is about start,FLContextKey.WORKFLOW,int,X,
    END_WORKFLOW,Workflow is about to end,FLContextKey.WORKFLOW,int,X,
    BEFORE_PROCESS_SUBMISSION,Task result submission is about to be processed,FLContextKey.TASK_NAME,str,X,
    ,,FLContextKey.TASK_RESULT,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    AFTER_PROCESS_SUBMISSION,Task result processing is done,FLContextKey.TASK_NAME,str,X,
    ,,FLContextKey.TASK_RESULT,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    BEFORE_TASK_DATA_FILTER,task data is about to be filtered,FLContextKey.TASK_NAME,str,X,X
    ,,FLContextKey.TASK_DATA,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    AFTER_TASK_DATA_FILTER,task data has been filtered,FLContextKey.TASK_NAME,str,X,X
    ,,FLContextKey.TASK_DATA,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    BEFORE_TASK_RESULT_FILTER,task result is about to be filtered,FLContextKey.TASK_NAME,str,X,X
    ,,FLContextKey.TASK_RESULT,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    AFTER_TASK_RESULT_FILTER,task result has been filtered,FLContextKey.TASK_NAME,str,X,X
    ,,FLContextKey.TASK_RESULT,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    BEFORE_TASK_EXECUTION,task execution is about to start,FLContextKey.TASK_NAME,str,,X
    ,,FLContextKey.TASK_DATA,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    AFTER_TASK_EXECUTION,task execution is has finished,FLContextKey.TASK_NAME,str,,X
    ,,FLContextKey.TASK_DATA,Shareable,,
    ,,FLContextKey.TASK_RESULT,Shareable,,
    ,,FLContextKey.TASK_ID,str,,
    BEFORE_SEND_TASK_RESULT,task result is about to be sent to the Server,FLContextKey.TASK_NAME,,,X
    ,,FLContextKey.TASK_DATA,,,
    ,,FLContextKey.TASK_RESULT,,,
    ,,FLContextKey.TASK_ID,,,
    AFTER_SEND_TASK_RESULT,task result has been sent to the Server,FLContextKey.TASK_NAME,,,X
    ,,FLContextKey.TASK_RESULT,,,
    ,,FLContextKey.TASK_DATA,,,
    ,,FLContextKey.TASK_ID,,,
    FATAL_SYSTEM_ERROR,fatal error occurred; the RUN is to be aborted,FLContextKey.EVENT_DATA,str; the error text,X,X
    FATAL_TASK_ERROR,fatal error in task execution; the task is to be aborted,FLContextKey.EVENT_DATA,str; the error text,,X
    ERROR_LOG_AVAILABLE,error log message available,FLContextKey.EVENT_DATA,str; the log message,X,X
    EXCEPTION_LOG_AVAILABLE,exception log message available,FLContextKey.EVENT_DATA,str; the log message,X,X

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

Local Events and Fed Events
---------------------------
Local events are local to the client, and federated events or fed events are broadcast to other sites as well.

The :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` widget will convert
local events to federated events.

One example of how this is applied in use is in log streaming as seen in the `hello-pt-tb example <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-pt-tb>`_.

The :class:`AnalyticsSender<nvflare.app_common.widgets.streaming.AnalyticsSender>` triggers an event called "analytix_log_stats",
as a local event on the client. If we want server side to receive this event, we will need to convert the local event
to a federated event, and this can be done with the :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` widget.

The :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` widget converts the event
to a federated event and adds a prefix to that event, which in the example becomes "fed.analytix_log_stats". This event
is processed by the :class:`TBAnalyticsReceiver<nvflare.app_common.pt.tb_receiver.TBAnalyticsReceiver>` component
on the server so the server can receive the streamed analytics.
