Federated XGBoost
=================


Overview
--------

NVFlare supports federated learning using popular gradient boosting library XGBoost.
It uses XGBoost library with federated plugin to perform the learning.

Following components are provided to run XGBoost jobs in NVFlare,

* :code:`nvflare.app_opt.xgboost.controller.XGBFedController`: The controller
  that starts the XGBoost federated server and kicks off all the XGBoost job on
  each NVFlare client. The configuration is generic for this component and
  no modification is needed for most training jobs.
* :code:`nvflare.app_opt.xgboost.executor.XGBExecutor`: This is the executor
  running on each NVFlare client, which starts XGBoost training. The
  configuration for this component needs to be customized for every site and for
  each training job because it contains job-specific parameters like location
  of training data.

Using XGBoost with NVFlare has following benefits compared with running federated XGBoost directly,

* XGBoost instance's life-cycle is managed by NVFlare. Both XGBoost client and server
  are started/stopped automatically by NVFlare workflow.
* XGBoost federated server can be configured automatically with auto-assigned port number.
* When mutual TLS is used, the certificates are managed by NVFlare using existing
  provisioning process.
* No need to manually configure each instance. Instance specific parameters
  like code:`rank` are assigned automatically by the NVFlare controller.

Requirements
------------

The XGBoost library with federated plugin must be installed on all the sites involved
in the learning.

Following the instructions here to build the XGBoost library with federated plugin,

https://github.com/dmlc/xgboost/tree/master/plugin/federated#readme

The Python package for XGBoost is also required. It can be installed using pip,
::
    pip install xgboost

Usage
-----

Basic components to run XGBoost are already included with NVFlare distribution.
Most XGBoost jobs can be created without custom code.

Please refer to :code:`NVFlare/examples/hello-xgboost` for an example
of a simple XGBoost learning job.

The server workflow is the same for all jobs, so the server configuration can be used
as is without modification. The default configuration starts the XGBoost federated
server on a random port. If a particular port (e.g. 4321) is required, it can be
configured as following,
::

    {
      "format_version": 2,
      "server": {
        "heart_beat_timeout": 600
      },
      "task_data_filters": [],
      "task_result_filters": [],
      "components": [],
      "workflows": [
        {
          "id": "xgb_controller",
          "path": "nvflare.app_opt.xgboost.controller.XGBFedController",
          "args": {
            "train_timeout": 30000,
            "port": 3456
          }
        }
      ]
    }

The client configuration uses an executor to run XGBoost. For example,
::

    {
      "format_version": 2,
      "data_root": "/dataset/",
      "components": [],
      "executors": [
        {
          "tasks": [
            "train"
          ],
          "executor": {
            "id": "Executor",
            "path": "nvflare.app_opt.xgboost.executor.XGBExecutor",
            "args": {
              "train_data": "{data_root}higgs.train.csv.1?format=csv&label_column=0",
              "test_data": "{data_root}higgs.test.csv.1?format=csv&label_column=0",
              "num_rounds": 100,
              "early_stopping_round": 2,
              "xgboost_params": {
                "max_depth": 8,
                "eta": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "nthread": 16
              }
            }
          }
        }
      ],

      "task_result_filters": [],
      "task_data_filters": []
    }

These parameters need to be adjusted for each learning job and for each site.
Most parameters are self-explanatory. Here are descriptions for a
few commonly used ones,

* :code:`train_data`: Location of the local training data.
  This is directly fed into DMatrix so it can be in any format
  that's supported by DMatrix.
* :code:`test_data`: Location of the local test data for verification.
  Also in DMatrix format.
* :code:`xgboost_params`: This dict is passed to :code:`xgboost.train()` as the first
  argument :code:`params`. It contains all the Booster parameters.
  Please refer to XGBoost documentation for details:
  https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training


GPU Support
-----------

If the CUDA is installed on the site, tree construction and prediction can be
accelerated using GPUs.

GPUs are enabled by using :code:`gpu_hist` as :code:`tree_method` parameter.
For example,
::
              "xgboost_params": {
                "max_depth": 8,
                "eta": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "gpu_hist",
                "gpu_id": 0,
                "nthread": 16
              }

Multiple GPUs can be supported by running one NVFlare client for each GPU. Each client
runs a different NVFlare app with the corresponding :code:`gpu_id` assigned.

Assuming there are 2 client sites, each with 2 GPUs (id 0 and 1), 4 NVFlare
client sites are needed, one for each GPU. The job layout looks like this,
::

    xgb_multi_gpu_job
    ├── app_server
    │   └── config
    │       └── config_fed_server.json
    ├── app_site1_gpu0
    │   └── config
    │       └── config_fed_client.json
    ├── app_site1_gpu1
    │   └── config
    │       └── config_fed_client.json
    ├── app_site2_gpu0
    │   └── config
    │       └── config_fed_client.json
    ├── app_site2_gpu1
    │   └── config
    │       └── config_fed_client.json
    └── meta.json

Each app is deployed to its own client site. Here is the :code:`meta.json`,
::

    {
      "name": "xgb_multi_gpu_job",
      "resource_spec": {
        "site-1a": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        },
        "site-1b": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        },
        "site-2a": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        },
        "site-2b": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        }
      },
      "deploy_map": {
        "app_server": [
          "server"
        ],
        "app_site1_gpu0": [
          "site-1a"
        ],
        "app_site1_gpu1": [
          "site-1b"
        ],
        "app_site2_gpu0": [
          "site-2a"
        ],
        "app_site2_gpu1": [
          "site-2b"
        ]
      },
      "min_clients": 4
    }

For federated XGBoost, all clients must participate in the training. There,
:code:`min_clients` must equal to the number of clients.

Customization
-------------

The provided XGBoost executor can be customized using Boost parameters
provided in :code:`xgboost_params` argument.

If the parameter change alone is not sufficient and code changes are required,
a custom executor can be implemented to make calls to xgboost library directly.

The executor must inherit the base class :code:`XGBExecutorBase` and implement
the :code:`xgb_train()` method.

For example, following custom executor can be used if a particular objective
function is required,
::

    class CustomXGBExecutor(XGBExecutorBase):
        def xgb_train(self, params: XGBoostParams, fl_ctx: FLContext) -> Shareable:
            with xgb.collective.CommunicatorContext(**params.communicator_env):
                dtrain = xgb.DMatrix(params.train_data)
                dtest = xgb.DMatrix(params.test_data)
                watchlist = [(dtest, "eval"), (dtrain, "train")]
                bst = xgb.train(
                    params.xgb_params,
                    dtrain,
                    params.num_rounds,
                    evals=watchlist,
                    early_stopping_rounds=params.early_stopping_rounds,
                    verbose_eval=params.verbose_eval,
                    callbacks=[callback.EvaluationMonitor(rank=self.rank)],
                    obj=squared_log,
                )

                # Save the model.
                workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
                run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
                run_dir = workspace.get_run_dir(run_number)
                bst.save_model(os.path.join(run_dir, "test.model.json"))
                xgb.collective.communicator_print("Finished training\n")

                return make_reply(ReturnCode.OK)

In the above example, :code:`squared_log` function is used as the objective
function, instead of the default one.