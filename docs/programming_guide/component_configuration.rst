.. _component_configuration:

**************************************************
NVFLARE Component Configuration and Event Handling
**************************************************

NVFLARE has a powerful configuration mechanism that can dynamically construct any components defined in the configurations.
In this section, we will discuss how NVFLARE configuration works, how to develop a new component and have that component be
recognized by NVFLARE and auto-registered with NVFLARE events. 

Background
==========
When a job is submitted, the system will deploy the job to the FL server and FL clients based on the deploy-map configuration.
The FL server and clients will parse the job configurations (fed_server_config.json and fed_client_config.json) respectively.  While the
configuration is parsed, the system will also dynamically construct the python objects based on the configuration. For each FLComponent
instantiated, it will also register the FLComponent into FLARE's event loop. 

This mechanism is very powerful; you can define a custom class extending FLComponent and then register the FLComponent in
the configuration file (fed_server_config.json or fed_client_config.json) and expect the component to be loaded into the FLARE system.  

Once the component is loaded, you can find it by ``component_id``, which is specified by you in the configuration file. 

Component configuration and lookup
==================================
To understand component configuration, we can look at the job configuration and see how the components are defined and
used. Below is the server side configuration for :ref:`hello_pt_job_api`.

.. code-block:: json

    {
        //<some lines skipped>
        "components": [
            {
                "id": "persistor",
                "path": "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor",
                "args": {}
            },
            {
                "id": "shareable_generator",
                "path": "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator",
                "args": {}
            },
            {
                "id": "aggregator",
                "path": "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
                "args": {
                    "expected_data_kind": "WEIGHTS"
                }
            },
            {
                "id": "model_locator",
                "path": "pt_model_locator.PTModelLocator",
                "args": {}
            },
            //<some lines skipped>
        ],
        "workflows": [
            {
                "id": "pre_train",
                "name": "InitializeGlobalWeights",
                "args": {
                    "task_name": "get_weights"
                }
            },
            {
                "id": "scatter_and_gather",
                "name": "ScatterAndGather",
                "args": {
                    "min_clients": 2,
                    "num_rounds": 2,
                    "start_round": 0,
                    "wait_time_after_min_received": 10,
                    "aggregator_id": "aggregator",
                    "persistor_id": "persistor",
                    "shareable_generator_id": "shareable_generator",
                    "train_task_name": "train",
                    "train_timeout": 0
                }
            },
            {
                "id": "cross_site_validate",
                "name": "CrossSiteModelEval",
                "args": {
                    "model_locator_id": "model_locator"
                }
            }
        ]
    }

Note the two sections for components and workflows.

Component Configuration
-----------------------
A FLARE job configuration defines a list of components. Here, we skip many other components so we can focus on just one component:

.. code-block:: json

    {
        "id": "aggregator",
        "path": "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
        "args": {
            "expected_data_kind": "WEIGHTS"
        }
    },

The component configuration consists of three parts:
    - component id: for example ``"id": "aggregator"``
    - component path, the fully qualified class path, for example: ``"path": "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",``
    - Component arguments, for example: ``"args": {"expected_data_kind": "WEIGHTS"}``

If we look at this class definition, we will find that this configuration is actually mapped to the class constructor:

.. code-block:: python

    class InTimeAccumulateWeightedAggregator(Aggregator):

        def __init__(
            self,
            exclude_vars: Union[str, Dict[str, str], None] = None,
            aggregation_weights: Union[Dict[str, Any], Dict[str, Dict[str, Any]], None] = None,
            expected_data_kind: Union[DataKind, Dict[str, DataKind]] = DataKind.WEIGHT_DIFF,
        ):

Notice the class takes 3 arguments: exclude_vars, aggregation_weights, and expected_data_kind. All of them have default values.

The above configuration essentially asks the system to instantiate the class using one argument, the other two arguments will use default values.

.. code-block:: python

    a = InTimeAccumulateWeightedAggregator(expected_data_kind = "WEIGHTS")

``config_type`` for Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some cases, you need to pass the arguments to the component as a dictionary, not as arguments of the constructor. The config_type to helps to specify the type. 

For example:

.. code-block:: json

    {
        "id": "shareable_generator",
        "path": "nvflare.app_opt.pt.fedopt.PTFedOptModelShareableGenerator",
        "args": {
            "device": "cpu",
            "source_model": "model",
            "optimizer_args": {
                "path": "torch.optim.SGD",
                "args": {
                    "lr": 1.0,
                    "momentum": 0.6
                },
                "config_type": "dict"
            },
            "lr_scheduler_args": {
                "path": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "args": {
                    "T_max": "{num_rounds}",
                    "eta_min": 0.9
                }
            }
        }
    },

Notice the config:

.. code-block:: json

    "optimizer_args": {
        "path": "torch.optim.SGD",
        "args": {
            "lr": 1.0,
            "momentum": 0.6
        },
        "config_type": "dict"
    },

We need to pass a run-time argument to "torch.optim.SDG" with a dictionary. To help the configuration parser to know that here we intend to pass a single dictionary
argument, not as two arguments to the constructor, we specify:

.. code-block:: json

    "config_type": "dict"

By default ``config_type`` is "Component" if not specified.

Name and Path
-------------
The class path can be quite long, so NVFLARE allows users to only specify the class name, and NVFLARE will search the specified Python path
to find the corresponding class path. In the configuration, you can use "name" to do this.

The configuration::

    "path": "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"

can be changed to::

    "name" : "InTimeAccumulateWeightedAggregator"

.. note::

    The class name must be in the $PYTHON_PATH in order for NVFLARE to find it. NVFlare built-in classes are all in the $PYTHON_PATH by default.

Looking up the component
^^^^^^^^^^^^^^^^^^^^^^^^
Once a component is registered, it can be accessed through the component_id, in the case of the example above: "id": "aggregator". 

To find the component, the runtime engine can be used. Assuming fl_ctx is the FL_Context object, you can get the component with the following:

.. code-block:: python

    engine = fl_ctx.get_engine()
    component = engine.get_component(component_id)

Failure Scenarios
^^^^^^^^^^^^^^^^^
Since the system dynamically instantiates the class based on configuration, there are cases where the class instantiation could fail, for example,
if args are required but not provided or if the constructor throws an exception.

When such a case happens, although the failure is class instantiation, FLARE may report the error as a configuration error since the class instantiation
failure originated from configuration parsing. You will need to look at the traceback and find the root cause of the failure.

Workflow Configuration
----------------------
The second part of the Job configuration is the workflow configuration with the key ``workflows``.

Workflows define a list of workflows. In the example above, three workflows are defined:

    - InitializeGlobalWeights for pre_train
    - ScatterAndGather for training with scatter_and_gatter 
    - CrossSiteModelEval for validation with cross_site_validate

Each workflow corresponds to a special type of FLComponent (known as a :ref:`Controller <controllers>`), which has the same component structure with an "id",
"name" (or "path"), and arguments that match the class definitions.

The controller arguments can be primitive types (int, str, etc.), or another component id.

Looking at the validation workflow, CrossSiteModelEval requires "model_locator_id". The value of "model_locator_id" is "model_locator", which is specified as
the id of one of the components defined in the configuration.  

Filters Configuration
^^^^^^^^^^^^^^^^^^^^^
There are additional optional filters such as ``task_data_filters`` or ``task_result_filters``. These correspond to the :ref:`filters` mechanism.

Component events
================
After understanding that components are instantiated dynamically based on the component configuration, another important aspect of
components is event handling.

NVIDIA FLARE comes with a powerful event mechanism that allows dynamic notifications to be sent to all objects that are of a subclass of
:ref:`fl_component`. To better understand the NVFLARE event system, see :ref:`event_system`. 

Examples of system events include::

    SYSTEM_START, 
    SYSTEM_END, 
    ABOUT_TO_START_RUN, 
    START_RUN, 
    ABOUT_TO_END_RUN
    END_RUN
    START_WORKFLOW
    END_WORKFLOW
    ABORT_TASK
    JOB_DEPLOYED
    JOB_STARTED
    JOB_COMPLETED
    JOB_ABORTED
    JOB_CANCELLED

.. note::

    This is not an exhaustive list of all events.

For federated learning applications, there are many application level events defined and fired. Here are some examples: 

    BEFORE_AGGREGATION
    END_AGGREGATION

    BEFORE_INITIALIZE
    AFTER_INITIALIZE
    BEFORE_TRAIN
    BEFORE_TRAIN_TASK
    AFTER_TRAIN 
    TRAINING_STARTED
    TRAINING_FINISHED
    TRAIN_DONE

    LOCAL_BEST_MODEL_AVAILABLE 
    GLOBAL_BEST_MODEL_AVAILABLE

    BEFORE_VALIDATE_MODEL 
    AFTER_VALIDATE_MODEL 

    ROUND_STARTED
    ROUND_DONE 

    INITIAL_MODEL_LOADED

    AFTER_AGGREGATION 
    GLOBAL_WEIGHTS_UPDATED

    CROSS_VAL_INIT 
    RECEIVE_BEST_MODEL


Each FLComponent will receive certain system events and application events, depending on if the component is a
Server or Client component. The FLComponent class can decide to handle or ignore the events.

Component configuration and event handling
==========================================
The second approach in component configuration: register components to handle events. 

Unlike the previous approach of component configuration, where we define a component in the job configuration,
then use the engine to lookup the component using component_id.  In this new approach,  the component Id is actually not
important, and most likely not used. 

All we need is to define an FLComponent, which will handle the specified event. There is no direct lookup of the
component. FLComponent will do its job in the event handle as long as the component is loaded into the system.

As we know from the previous section, loading components into the system can be accomplished by simply adding the
components configuration in the job configuration file.

The only decision you have to make is to decide where the component should be placed: on server side ( fed_server_config.json)
or client side (fed_client_config.json).

Here is one concrete example of such a mechanism. In many of NVFLARE examples, you might have noticed that the job components has::

    {
        "id": "model_selector",
        "name": "IntimeModelSelector",
        "args": {}
    }

:class:`nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector` is an FLComponent designed for selecting
the best global model to save, usually associated
with a "validate" task. IntimeModelSelector handles application events and selects the best model based on validation
scores sent back from the clients. If you want to leverage this model selection mechanism, all you needs to do is add
this component to the server job component configuration (config code shown above).

.. code-block:: python

    class IntimeModelSelector(Widget):
    
        ...
    
        def handle_event(self, event_type: str, fl_ctx: FLContext):
            if event_type == EventType.START_RUN:
                self._startup()
            elif event_type == AppEventType.ROUND_STARTED:
                self._reset_stats()
            elif event_type == AppEventType.BEFORE_CONTRIBUTION_ACCEPT:
                self._before_accept(fl_ctx)
            elif event_type == AppEventType.BEFORE_AGGREGATION:
                self._before_aggregate(fl_ctx)
    
        ...
    
        def _before_aggregate(self, fl_ctx):
    
        ...
    
            if self.val_metric > self.best_val_metric:
                self.best_val_metric = self.val_metric
        
            ...
            
                # Fire event to notify that the current global model is a new best
                self.fire_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)
    
    ...

Notice that when IntimeModelSelector handles ``BEFORE_AGGREGATION`` event, once it found the best model, it will simply
fire another application event: ``AppEventType.GLOBAL_BEST_MODEL_AVAILABLE``. 

Another FLComponent responsible for performing persistence (persistor) will listen to the event ``GLOBAL_BEST_MODEL_AVAILABLE``,
then can retrieve and save the model to a storage location. 

If you decided to write a different model selector based on different criteria or different event, all you need to do
is write a new FLComponent (subclass IntimeModelSelector or simply write one from scratch), then add your component to
the job configuration.
