.. _cross_site_model_evaluation:

Cross Site Model Evaluation / Federated Evaluation
--------------------------------------------------
The :class:`cross site model evaluation workflow<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`
uses the data from clients to run evaluation with the models of other clients.
Data is not shared, rather the collection of models is distributed to each client site to run local validation.  The
results of local validation are collected by the server to construct an all-to-all matrix of model performance vs.
client dataset.

The server’s global model is also distributed to each client for evaluation on the client’s local dataset for global
model evaluation.

The `hello-numpy-cross-val example <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-numpy-cross-val>`_ is a simple
example that implements the :class:`cross site model evaluation workflow<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`.

.. note::

   Previously in NVFlare before version 2.0, cross-site validation was built into the framework itself, and there was an
   admin command to retrieve cross-site validation results. In NVFlare 2.0, with the ability to have customized
   workflows, cross-site validation is no longer in the NVFlare framework but is instead handled by the workflow. The
   the `cifar10 example <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_ is configured to run cross-site
   model evaluation and ``config_fed_server.json`` is configured with :class:`ValidationJsonGenerator<nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator>`
   to write the results to a JSON file on the server.
