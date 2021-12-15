.. _cross_site_model_evaluation:

Cross Site Model Evaluation / Federated Evaluation
--------------------------------------------------
The cross site model evaluation workflow uses the data from clients to run evaluation with the models of other clients.
Data is not shared, rather the collection of models is distributed to each client site to run local validation.  The
results of local validation are collected by the server to construct an all-to-all matrix of model performance vs.
client dataset.

The server’s global model is also distributed to each client for evaluation on the client’s local dataset for global
model evaluation.

See the code for details:

:class:`nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval`
