.. _hello_pt:

Hello PyTorch
=============

Hello PyTorch is the recommended first federated-learning example for PyTorch.
It uses :class:`FedAvgRecipe<nvflare.app_opt.pt.recipes.fedavg.FedAvgRecipe>`
with ordinary PyTorch model, data-loading, training, and evaluation code.

The zero-argument path is deterministic, CPU-safe, and offline. The default
runs three federated rounds across two simulated clients using independently
generated synthetic image data, then evaluates the persisted final global model
on separate site-local evaluation data. CIFAR-10 remains an explicit follow-up
option.

The :github_nvflare_link:`example README <examples/hello-world/hello-pt/README.md>`
is the authoritative reference for every option, default, artifact, and
troubleshooting note. This page provides the guided first-run path.

Get and install the example
---------------------------

Create and activate a Python virtual environment, then get the source and enter
the example directory:

.. code-block:: bash

   git clone https://github.com/NVIDIA/NVFlare.git
   cd NVFlare/examples/hello-world/hello-pt

Install the dependencies from that directory:

.. code-block:: bash

   python -m pip install -r requirements.txt

For alternative installation methods, see :ref:`installation`.

Run the quickstart
------------------

.. code-block:: bash

   python job.py

The default run uses two simulated clients, three federated rounds, one local
epoch per round, and no data download or tracking service. Each client receives
reproducible samples generated independently from the same simple IID
distribution. Labels are encoded by class-specific image regions, giving the
small convolutional network a genuine and testable learning signal instead of
unrelated random images and labels. This quickstart does not claim to model
statistical heterogeneity.

The client script follows the Client API lifecycle:

1. Receive the current global model.
2. Evaluate that received model.
3. Train it on the client's local data.
4. Send updated model parameters, metrics, and completed optimizer-step count.

Raw examples remain at the client. The server performs weighted FedAvg
aggregation, persists the final global model, and requests its final evaluation
on both sites.

Inspect the result
------------------

The command prints the result directory. For the default simulation it is
``/tmp/nvflare/simulation/hello-pt``. The primary artifacts under
``server/simulate_job`` are:

- ``app_server/FL_global_model.pt`` -- the persisted final global model.
- ``metrics/metrics_summary.json`` -- final aggregated training-round metrics
  and available best-model metric metadata.
- ``cross_site_val/cross_val_results.json`` -- post-training evaluation of the
  persisted final model by site.

Use ``metrics_summary.json`` for a compact summary of the federated training
metrics. To inspect the accuracy of the persisted model after the last
aggregation, use each site's ``SRV_FL_global_model.pt`` entry in
``cross_val_results.json``. These values can differ because clients report
training-round accuracy before local training and the final aggregation occurs
after the last such report.

The automated acceptance test requires at least 60% final accuracy on both
sites and at least a 40 percentage-point improvement over the initial global
model. These thresholds are calibrated to the fixed model and data seeds with
the three-round default. They verify this specific run's learning signal, not
arbitrary initializations or hyperparameters, and are not benchmark claims.

Export the application
----------------------

You can export the application without running the simulation:

.. code-block:: bash

   python job.py --export --export-dir /tmp/nvflare/jobs/job_config

The shared Recipe layer reports that it consumes these system-level arguments
before the example parser. Export verifies construction of the deployable job;
it does not verify production connectivity, identity, authorization, or
execution.

Optional follow-up paths
------------------------

Run ``python job.py --help`` for the complete example and Recipe export options.
CIFAR-10 is also available through ``--dataset cifar10``. Run
``python prepare_data.py`` first to download both splits before simulated
clients open the shared cache. All clients then read the same logical CIFAR-10
datasets, so this optional path does not demonstrate a federated data
partition. For a non-default cache, pass the same ``--data_root`` value to
``prepare_data.py`` and ``job.py``. The example README provides the exact
commands.

The beginner entry point intentionally exposes only client count, round count,
dataset choice, and the client-local data root. Environment selection,
experiment tracking, full cross-site evaluation, external-process execution,
and memory tuning belong in a separate continuation workflow rather than the
first federated-learning run.

Continue to POC and Production
------------------------------

After completing the simulation, continue with the
:github_nvflare_link:`advanced environment-continuity example
<examples/advanced/hello-pt-environments/README.md>` to run the same learning
application in a local POC or an already-running production deployment.

For the API concepts behind the example, continue with
:ref:`Client API <client_api>` and :ref:`Available Recipes <available_recipes>`.
