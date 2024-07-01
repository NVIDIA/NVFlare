##############################
Federated XGBoost with NVFlare
##############################

XGBoost (https://github.com/dmlc/xgboost) is an open-source project that
implements machine learning algorithms under the Gradient Boosting framework.
It is an optimized distributed gradient boosting library designed to be highly
efficient, flexible and portable.
This implementation uses MPI (message passing interface) for client
communication and synchronization.

MPI requires the underlying communication network to be perfect - a single
message drop causes the training to fail.

This is usually achieved via a highly reliable special-purpose network like NCCL.

The open-source XGBoost supports federated paradigm, where clients are in different
locations and communicate with each other with gRPC over internet connections.

We introduce federated XGBoost with NVFlare for a more reliable federated setup.

.. toctree::
   :maxdepth: 1

   federated_xgboost/reliable_xgboost_design
   federated_xgboost/reliable_xgboost_timeout
   federated_xgboost/secure_xgboost_design
   federated_xgboost/secure_xgboost_user_guide
