.. _federated_xgboost:

Federated XGBoost
=================

Overview
--------

NVFlare supports federated learning using popular gradient boosting library XGBoost.
It uses XGBoost library with federated plugin (xgboost version >= 1.7.0rc1) to perform the learning.

Using XGBoost with NVFlare has following benefits compared with running federated XGBoost directly,

* XGBoost instance's life-cycle is managed by NVFlare. Both XGBoost client and server
  are started/stopped automatically by NVFlare workflow.
* For histogram-based XGBoost federated server can be configured automatically with auto-assigned port number.
* When mutual TLS is used, the certificates are managed by NVFlare using existing
  provisioning process.
* No need to manually configure each instance. Instance specific parameters
  like code:`rank` are assigned automatically by the NVFlare controller.

Examples
--------

Basic components to run XGBoost are already included with NVFlare distribution.
Most XGBoost jobs can be created without custom code.

Please refer to :code:`NVFlare/examples/advanced/xgboost` for more details.

Previous Versions of Federated XGBoost
--------------------------------------

   - `Federated XGBoost for 2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/xgboost>`_
