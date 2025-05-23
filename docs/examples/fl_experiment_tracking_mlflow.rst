.. _experiment_tracking_mlflow:

FL Experiment Tracking with MLflow
==================================

Introduction
-------------

The example for experiment tracking with MLflow has clients streaming their statistics to the server through
events and the server writing the statistics to MLflow. This is similar to the :ref:`tensorboard_streaming` example
but uses MLflow as a back end for experiment tracking. This example is in the advanced examples folder under 
experiment-tracking, in the "mlflow" directory.

The setup of this exercise consists of one **server** and two **clients**. The clients stream their statistics to
the server as events with :class:`MLflowWriter<nvflare.app_opt.tracking.mlflow.mlflow_writer.MLflowWriter>`,
and only the server writes data to the MLflow tracking server with
:class:`MLflowReceiver<nvflare.app_opt.tracking.mlflow.mlflow_receiver.MLflowReceiver>`. This allows the server to
be the only party that needs to deal with authentication and communication with the MLflow tracking server, and
streamlines and reduces the communication by buffering the data to send.

.. note::

  Like :ref:`tensorboard_streaming`, this exercise differs from :doc:`hello_pt` by using the ``Learner`` API along with the ``LearnerExecutor``.
  In short, the execution flow is abstracted away into the ``LearnerExecutor``, allowing you to only need to implement the required methods in the ``Learner`` class.
  This will not be the focus of this guide, however you can learn more at :class:`Learner<nvflare.app_common.abstract.learner_spec.Learner>`
  and :class:`LearnerExecutor<nvflare.app_common.executors.learner_executor.LearnerExecutor>`.


Let's get started. Make sure you have an environment with NVIDIA FLARE installed as described in
:ref:`getting_started`. First clone the repo:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Now remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.

Install the required dependencies (NVFlare/examples/advanced/experiment-tracking/mlflow).

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install -r requirements.txt

When running, make sure to set `PYTHONPATH` to include the custom files of the example (replacing the path below
with the appropriate path to the directory containing the "pt" directory with custom files):

.. code-block:: shell

  (nvflare-env) $ export PYTHONPATH=${YOUR PATH TO NVFLARE}/examples/advanced/experiment-tracking

Adding MLflow Logging to Configurations
------------------------------------------------

Inside the config folder there are two files, ``config_fed_client.conf`` and ``config_fed_server.conf``.

.. literalinclude:: ../../examples/advanced/experiment-tracking/mlflow/jobs/hello-pt-mlflow/app/config/config_fed_client.conf
   :caption: config_fed_client.conf

Take a look at the components section of the client config at line 24.
The first component is the ``pt_learner`` which contains the initialization, training, and validation logic.
``learner_with_mlflow.py`` (under NVFlare/examples/advanced/experiment-tracking/pt) contains the code written for the MLflowWriter syntax.

The :class:`MLflowWriter<nvflare.app_opt.tracking.mlflow.mlflow_writer.MLflowWriter>` mimics the syntax of mlflow, to make it easier to use existing code
that is using MLflow for metrics tracking. Instead of writing to the MLflow tracking server, however, the MLflowWriter creates and sends an event
within NVFlare with the information to track.

Finally, :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` converts local events to federated events.
This changes the event ``analytix_log_stats`` into a fed event ``fed.analytix_log_stats``, which will then be streamed from the clients to the server.

.. literalinclude:: ../../examples/advanced/experiment-tracking/mlflow/jobs/hello-pt-mlflow/app/config/config_fed_server.conf
   :caption: config_fed_server.conf

Under the component section in the server config, we have the
:class:`MLflowReceiver<nvflare.app_opt.tracking.mlflow.mlflow_receiver.MLflowReceiver>`. This component receives
events from the clients and internally buffers them before writing to the MLflow tracking server. The default
"buffer_flush_time" is one second, but this can be configured as an arg in the component config for MLflowReceiver.

Notice how the accepted event type ``"fed.analytix_log_stats"`` matches the output of
:class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` in the client config.


Adding MLflow Logging to Your Code
-------------------------------------------

In this exercise, all of the MLflow code additions will be made in ``learner_with_mlflow.py``.

First we must initialize our MLflow writer we defined in the client config:

.. literalinclude:: ../../examples/advanced/experiment-tracking/pt/learner_with_mlflow.py
   :language: python
   :lines: 102-105
   :lineno-start: 102
   :linenos:

The ``LearnerExecutor`` passes in the component dictionary into the ``parts`` parameter of ``initialize()``.
We can then access the ``MLflowWriter`` component we defined in ``config_fed_client.json``
by using the ``self.analytic_sender_id`` as the key in the ``parts`` dictionary.
Note that ``self.analytic_sender_id`` defaults to ``"analytic_sender"``,
but we can also define it in the client config to be passed into the constructor.

Now that our writer is set to ``MLflowWriter``,
we can write and stream training metrics to the server in ``local_train()``:

.. literalinclude:: ../../examples/advanced/experiment-tracking/pt/learner_with_mlflow.py
   :language: python
   :lines: 148-182
   :lineno-start: 148
   :linenos:

We use ``self.writer.log_metrics()`` on line 178 to send training loss metrics,
while on line 182 we send the validation accuracy at the end of each epoch.

You can see the currently supported methods for MLflowWriter in
:class:`MLflowWriter<nvflare.app_opt.tracking.mlflow.mlflow_writer.MLflowWriter>`.


Train the Model, Federated!
---------------------------

.. |ExampleApp| replace:: hello-pt-mlflow
.. include:: run_fl_system.rst


Viewing the MLflow UI
---------------------------------
By default, MLflow will create an experiment log directory under a directory named "mlruns" in the
workspace. For example, if your server workspace is located at "/example_workspace/workspace/example_project/prod_00/server-1",
then you can launch the MLflow UI with:

.. code-block:: shell

   mlflow ui --backend-store-uri /example_workspace/workspace/example_project/prod_00/server-1


.. include:: access_result.rst

.. include:: shutdown_fl_system.rst

Congratulations!

Now you will be able to see the live training metrics of each client from MLflow, streamed from the server.

The full source code for this exercise can be found in
:github_nvflare_link:`examples/advanced/experiment-tracking/mlflow <examples/advanced/experiment-tracking/mlflow>`.
