.. _unsafe_component_detection:

**************************
Unsafe Component Detection
**************************
NVFLARE is based on a componentized architecture in that FL jobs are performed by components that are configured in configuration
files. These components are created at the beginning of job execution. To address the issue of components potentially being unsafe
and leaking sensitive information, NVFLARE uses an event based solution.

NVFLARE has a very powerful and flexible event mechanism that allows custom code to be plugged into defined moments of system
workflow (e.g. start/end of the job, before/after a task is executed, etc.). At such moments, NVFLARE fires events and invokes
:ref:`fl_component` objects that handle these events. 

The ``BEFORE_BUILD_COMPONENT`` event type can allow a custom FLComponent to detect unsafe job components during the time of configuration processing. This event
type is fired before the configuration processor starts to build a job component (executor, filter, etc.). 

Detect Unsafe Job Components
============================
To detect unsafe job components, the user simply needs to create a custom FLComponent object that handles this event,
as shown in the following ComponentChecker example:

.. code-block:: python

    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_component import FLComponent
    from nvflare.apis.fl_constant import FLContextKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.fl_exception import UnsafeComponentError

    class ComponentChecker(FLComponent):

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        prop_keys = fl_ctx.get_prop_keys()
        if event_type == EventType.BEFORE_BUILD_COMPONENT:
            print(f"ComponentChecker: fl_ctx props: {prop_keys}")
            comp_config = fl_ctx.get_prop(FLContextKey.COMPONENT_CONFIG)
            print(f"Comp Config: {comp_config}")
            raise UnsafeComponentError("client encountered bad component")


The important points are:

    - The class must extend FLComponent
    - It defines the handle_event method, following the exact signature
    - It checks if the event_type is ``EventType.BEFORE_BUILD_COMPONENT``. 
    - It checks the component being built based on the information provided in the fl_ctx. There are many properties in fl_ctx. The most important ones are the ``COMPONENT_CONFIG`` that is a dict of the component's configuration data. The fl_ctx also has ``WORKSPACE_OBJECT`` which allows access to any file in the job's workspace.
    - If any issue is detected with the component to be built, you raise the ``UnsafeComponentError`` exception with a meaningful text.

The following properties in the fl_ctx could be helpful too:

``FLContextKey.COMPONENT_NODE`` - This gives you the information about the component's location in the config structure (which could be viewed as a tree).

``FLContextKey.CONFIG_CTX`` - This gives you information about the entire config structure.

``FLContextKey.CURRENT_JOB_ID`` - The ID of the current job.

``FLContextKey.JOB_META`` - This is a dict that contains meta information (e.g. job submitter's name, org and role) about the current job.

``FLContextKey.WORKSPACE_OBJECT`` - This object provides many convenience methods to determine the paths of files in the workspace

Install Your Component Checker
==============================
Once you define your component checker (you can name your class any way you want - does not have to be ComponentChecker), you need
to install it to your FL site(s).

First of all, your custom code could be included as part of your FL docker, depending on how you manage the docker. If this is not
possible, then you can include it in the FL site's ``<workspace_root>/local/custom`` folder.

Second, include this custom component in your site's ``job_resources.json``, as shown here:

.. code-block:: json

    {
        "format_version": 2,
        "components": [
            {
                "id": "comp_checker",
                "path": "comp_auth.ComponentChecker"
            }
        ]
    }

Your site's workspace should look like this:

.. code-block::

    workspace_root
        local
            resources.json
            job_resources.json
            ...
            custom
                comp_auth.py
        startup
        ...

