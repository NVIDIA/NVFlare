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

The ``BEFORE_BUILD_COMPONENT`` event type can allow a custom FLComponent to detect unsafe job components during the time of
configuration processing. This event type is fired before the configuration processor starts to build a job component
(executor, filter, etc.). It is also fired for component configs that are nested recursively inside another component's
``args``.

Detect Unsafe Job Components
============================
To detect unsafe job components, the user simply needs to create a custom FLComponent object that handles this event,
as shown in the following ComponentChecker example. This example is intentionally minimal: it demonstrates the event
handling pattern and how to raise ``UnsafeComponentError`` when a problem is found. It is not a complete production
implementation of component safety policy.

.. code-block:: python

    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_component import FLComponent
    from nvflare.apis.fl_constant import FLContextKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.fl_exception import UnsafeComponentError

    class ComponentChecker(FLComponent):
        def handle_event(self, event_type: str, fl_ctx: FLContext):
            if event_type == EventType.BEFORE_BUILD_COMPONENT:
                comp_config = fl_ctx.get_prop(FLContextKey.COMPONENT_CONFIG)
                if "name" in comp_config:
                    raise UnsafeComponentError("component config must use path or class_path")
                elif "path" in comp_config:
                    component_path = comp_config["path"]
                elif "class_path" in comp_config:
                    component_path = comp_config["class_path"]
                else:
                    return
                if component_path == "bad_package.BadComponent":
                    raise UnsafeComponentError(f"component is not allowed: {component_path}")


The important points are:

    - The class must extend FLComponent
    - It defines the handle_event method, following the exact signature
    - It checks if the event_type is ``EventType.BEFORE_BUILD_COMPONENT``. 
    - It checks the component being built based on the information provided in the fl_ctx. There are many properties in fl_ctx. The most important ones are the ``COMPONENT_CONFIG`` that is a dict of the component's configuration data. The fl_ctx also has ``WORKSPACE_OBJECT`` which allows access to any file in the job's workspace.
    - If any issue is detected with the component to be built, you raise the ``UnsafeComponentError`` exception with a meaningful text.

The following properties in the fl_ctx could be helpful too:

``FLContextKey.COMPONENT_NODE`` - This gives you the information about the component's location in the config structure
(which could be viewed as a tree). For nested component configs inside ``args``, this path contains each nesting level,
for example ``component.args.child.args.worker``.

``FLContextKey.CONFIG_CTX`` - This gives you information about the entire config structure.

``FLContextKey.CURRENT_JOB_ID`` - The ID of the current job.

``FLContextKey.JOB_META`` - This is a dict that contains meta information (e.g. job submitter's name, org and role) about the current job.

``FLContextKey.WORKSPACE_OBJECT`` - This object provides many convenience methods to determine the paths of files in the workspace

Use the Built-in Component Path Authorizer
------------------------------------------
When BYOC is disabled, NVFLARE runs a built-in component path authorization check while parsing job
configuration. Sites get this protection without installing an authorizer component in ``resources.json``. The built-in
policy allows only class paths that match ``class_allow_list`` in the site's top-level ``resources.json`` or
``resources.json.default``. If ``class_allow_list`` is not configured, NVFLARE uses the curated built-in default shown
below. An explicitly configured list replaces that default.

Migration note for upgrades: startup kits created before this policy may not contain ``class_allow_list``. Such sites use
the built-in default automatically. Add a top-level ``class_allow_list`` to each site's ``resources.json`` or
``resources.json.default`` only when the site needs to replace the default, for example to authorize reviewed site-local
classes for non-BYOC jobs.

The check is applied to every component config built through the NVFLARE JSON configuration flow, including component configs
nested at any depth inside another component's ``args``. It also checks component configs inside dictionaries and lists before
they can be built later by runtime builders such as the multi-process executor or ``engine.build_component()``. The
multi-process executor's ``components`` entries are checked even if an entry sets ``"config_type": "dict"``, because those
entries are still built as components later. The authorizer can also be called directly with
``authorize_component_config(...)`` by code that wants to validate a component config without firing an event.

When BYOC is enabled for the job, this built-in class allow-list check is skipped because BYOC authorization already permits
loading job-provided custom code.

Under this policy, component configs must use either ``path`` or ``class_path`` as the fully qualified class path key.
If both are present, ``path`` takes precedence. Key presence is used, not truthiness: if ``path`` is present but empty or
invalid, it is rejected instead of falling through to ``class_path``. Component configs that include ``name`` are rejected
by the built-in path authorizer; non-BYOC jobs should use ``path`` or ``class_path`` so the fully qualified class path can
be checked against ``class_allow_list``.

``class_allow_list`` is a list of allowed component path prefixes. Package prefixes should end with ``.`` to match on a
Python package boundary, for example ``"nvflare."``. Entries without a trailing ``.`` must be fully qualified dotted paths
and are matched exactly or on a ``.`` boundary. For example, ``"nvflare"`` is rejected as ambiguous, and ``"nvflare."`` does
not match ``"nvflareevil.module.Component"``.

The adjacent ``class_list_enforcement_mode`` setting accepts ``"enforce"`` (the default) or ``"warn"``. In ``"warn"``
mode, a component outside ``class_allow_list`` is allowed to load and a warning is logged. If ``"*"`` appears anywhere in
``class_allow_list``, all component classes are allowed, the remaining entries are ignored, and an audit event records that
the allow-list check was bypassed. Use ``"warn"`` and ``"*"`` only as temporary migration aids in trusted environments.

Provisioned ``resources.json.default`` Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When provisioning generates startup kits, the server and client ``resources.json.default`` files include the following
top-level ``class_allow_list``. It is also the built-in fallback when the setting is omitted. Operators can replace it in
``resources.json`` or ``resources.json.default`` to match the classes their non-BYOC jobs are allowed to load.

.. code-block:: json

    {
        "format_version": 2,
        "class_list_enforcement_mode": "enforce",
        "class_allow_list": [
            "nvflare.app_common.aggregators.collect_and_assemble_model_aggregator.CollectAndAssembleModelAggregator",
            "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
            "nvflare.app_common.ccwf.comps.simple_model_shareable_generator.SimpleModelShareableGenerator",
            "nvflare.app_common.ccwf.cse_client_ctl.CrossSiteEvalClientController",
            "nvflare.app_common.ccwf.cse_server_ctl.CrossSiteEvalServerController",
            "nvflare.app_common.ccwf.cyclic_client_ctl.CyclicClientController",
            "nvflare.app_common.ccwf.cyclic_server_ctl.CyclicServerController",
            "nvflare.app_common.ccwf.swarm_client_ctl.SwarmClientController",
            "nvflare.app_common.ccwf.swarm_server_ctl.SwarmServerController",
            "nvflare.app_common.executors.statistics.statistics_executor.StatisticsExecutor",
            "nvflare.app_common.filters.statistics_privacy_filter.StatisticsPrivacyFilter",
            "nvflare.app_common.logging.job_log_receiver.JobLogReceiver",
            "nvflare.app_common.logging.job_log_streamer.JobLogStreamer",
            "nvflare.app_common.np.np_model_locator.NPModelLocator",
            "nvflare.app_common.np.np_model_persistor.NPModelPersistor",
            "nvflare.app_common.np.np_validator.NPValidator",
            "nvflare.app_common.psi.dh_psi.dh_psi_controller.DhPSIController",
            "nvflare.app_common.psi.file_psi_writer.FilePSIWriter",
            "nvflare.app_common.psi.psi_executor.PSIExecutor",
            "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator",
            "nvflare.app_common.statistics.histogram_bins_cleanser.HistogramBinsCleanser",
            "nvflare.app_common.statistics.json_stats_file_persistor.JsonStatsFileWriter",
            "nvflare.app_common.statistics.min_count_cleanser.MinCountCleanser",
            "nvflare.app_common.statistics.min_max_cleanser.AddNoiseToMinMax",
            "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector",
            "nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator",
            "nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval",
            "nvflare.app_common.workflows.cyclic_ctl.CyclicController",
            "nvflare.app_common.workflows.fedavg.FedAvg",
            "nvflare.app_common.workflows.lr.fedavg.FedAvgLR",
            "nvflare.app_common.workflows.lr.np_persistor.LRModelPersistor",
            "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather",
            "nvflare.app_common.workflows.scaffold.Scaffold",
            "nvflare.app_common.workflows.statistics_controller.StatisticsController",
            "nvflare.app_opt.he.intime_accumulate_model_aggregator.HEInTimeAccumulateWeightedAggregator",
            "nvflare.app_opt.he.model_decryptor.HEModelDecryptor",
            "nvflare.app_opt.he.model_encryptor.HEModelEncryptor",
            "nvflare.app_opt.he.model_serialize_filter.HEModelSerializeFilter",
            "nvflare.app_opt.he.model_shareable_generator.HEModelShareableGenerator",
            "nvflare.app_opt.psi.dh_psi.dh_psi_task_handler.DhPSITaskHandler",
            "nvflare.app_opt.pt.fedopt.PTFedOptModelShareableGenerator",
            "nvflare.app_opt.pt.file_model_locator.PTFileModelLocator",
            "nvflare.app_opt.pt.recipes.fedeval.EvalController",
            "nvflare.app_opt.sklearn.kmeans_assembler.KMeansAssembler",
            "nvflare.app_opt.sklearn.svm_assembler.SVMAssembler",
            "nvflare.app_opt.tf.fedopt_ctl.FedOpt",
            "nvflare.app_opt.tf.file_model_locator.TFFileModelLocator",
            "nvflare.app_opt.tracking.mlflow.mlflow_receiver.MLflowReceiver",
            "nvflare.app_opt.tracking.mlflow.mlflow_writer.MLflowWriter",
            "nvflare.app_opt.tracking.tb.tb_receiver.TBAnalyticsReceiver",
            "nvflare.app_opt.tracking.tb.tb_writer.TBWriter",
            "nvflare.app_opt.tracking.wandb.wandb_receiver.WandBReceiver",
            "nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader.CSVDataLoader",
            "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
            "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
            "nvflare.app_opt.xgboost.tree_based.bagging_aggregator.XGBBaggingAggregator",
            "nvflare.app_opt.xgboost.tree_based.executor.FedXGBTreeExecutor",
            "nvflare.app_opt.xgboost.tree_based.model_persistor.XGBModelPersistor",
            "nvflare.app_opt.xgboost.tree_based.shareable_generator.XGBModelShareableGenerator"
        ],
        "components": [
        ]
    }

With the policy above, a non-BYOC job component configured with ``"path": "subprocess.Popen"`` is rejected because it does
not match any entry in ``class_allow_list``. The same rule applies to ``"class_path": "subprocess.Popen"``.
The provisioned list intentionally excludes framework optimizer, scheduler, and model classes. If a job configures those
classes, each site must add the reviewed class paths or package prefixes to
``class_allow_list`` before running the job with BYOC disabled.

This is an allow-list baseline. It is not a replacement for secure job review, least-privilege runtime environments, container or
process sandboxing, and other controls appropriate to your deployment.

Install Your Component Checker
------------------------------
Once you define your component checker (you can name your class any way you want - does not have to be ComponentChecker), you need
to install it to your FL site(s).

First of all, your custom code could be included as part of your FL docker, depending on how you manage the docker. If this is not
possible, then you can include it in the FL site's ``<workspace_root>/local/custom`` folder.

Second, include this custom component in your site's ``resources.json``, as shown here:

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
            ...
            custom
                comp_auth.py
        startup
        ...
