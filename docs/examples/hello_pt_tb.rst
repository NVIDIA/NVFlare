Quickstart (PyTorch with TensorBoard)
=====================================

Introduction
-------------

In this exercise, you will learn how to stream TensorBoard events from the clients to the server in order to visualize live training metrics from a central place on the server.
This exercise will be working with the ``hello-pt-tb`` application in the examples folder, which builds upon :doc:`hello_pt` by adding TensorBoard streaming.

.. note::

  This exercise also differs from :doc:`hello_pt`, as it uses the ``Learner`` API along with the ``LearnerExecutor``.
  In short, the execution flow is abstracted away into the ``LearnerExecutor``, allowing you to only need to implement the required methods in the ``Learner`` class.
  This will not be the focus of this guide, however you can learn more at :class:`Learner<nvflare.app_common.abstract.learner_spec.Learner>`
  and :class:`LearnerExecutor<nvflare.app_common.executors.learner_executor.LearnerExecutor>`.


Let's get started. Make sure you have an environment with NVIDIA FLARE installed as described in
:doc:`installation <../installation>` guide. First clone the repo:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Now remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.
Since you will use PyTorch, torchvision, and TensorBoard for this exercise, let's go ahead and install these libraries:

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install torch torchvision tensorboard


Adding TensorBoard Streaming to Configurations
------------------------------------------------

Inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.

.. literalinclude:: ../../examples/hello-pt-tb/config/config_fed_client.json
   :language: json
   :linenos:
   :caption: config_fed_client.json

Take a look at the components section of the client config at line 24. The first component is the ``pt_learner`` which contains the initialization, training, and validation logic.
``pt_learner.py`` is where we will add our TensorBoard streaming changes.

Next we have the :class:`AnalyticsSender<nvflare.app_common.widgets.streaming.AnalyticsSender>`, which implements some common methods that follow the signatures from the PyTorch SummaryWriter.
This makes it easy for the ``pt_learner`` to log metrics and send events.

Finally, we have the :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>`, which converts local events to federated events.
This changes the event ``analytix_log_stats`` into a fed event ``fed.analytix_log_stats``, which will then be streamed from the clients to the server.

.. literalinclude:: ../../examples/hello-pt-tb/config/config_fed_server.json
   :language: json
   :linenos:
   :caption: config_fed_server.json

Under the component section in the server config, we have the ``TBAnalyticsReceiver`` of type :class:`AnalyticsReceiver<nvflare.app_common.widgets.streaming.AnalyticsReceiver>`.
This component receives TensorBoard events from the clients and saves them to a specified folder (default ``tb_events``) under the server's run folder. Notice how the accepted event type
``"fed.analytix_log_stats"`` matches the output of  :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` in the client config.


Adding TensorBoard Streaming to your Code
-------------------------------------------

In this exercise, all of the TensorBoard code additions will be made in ``pt_learner.py``.

First we must initalize our TensorBoard writer to the ``AnalyticsSender`` we defined in the client config:

.. literalinclude:: ../../examples/hello-pt-tb/custom/pt_learner.py
   :language: python
   :lines: 61, 89-92
   :lineno-start: 61
   :linenos:

The ``LearnerExecutor`` passes in the component dictionary into the ``parts`` parameter of ``initialize()``.
We can then access the ``AnalyticsSender`` component we defined in ``config_fed_client.json`` by using the ``self.analytic_sender_id`` as the key in the ``parts`` dictionary.
Note that ``self.analytic_sender_id`` defaults to ``"analytic_sender"``, but we can also define it in the client config to be passed into the constructor.

Now that our TensorBoard writer is set to ``AnalyticsSender``, we can write and stream training metrics to the server in ``local_train()``:

.. literalinclude:: ../../examples/hello-pt-tb/custom/pt_learner.py
   :language: python
   :lines: 127-159
   :lineno-start: 127
   :linenos:

We use ``add_scalar(tag, scalar, global_step)`` on line 155 to send training loss metrics, while on line 159 we send the validation accuracy at the end of each epoch.
You can learn more about other supported writer methods in :class:`AnalyticsSender<nvflare.app_common.widgets.streaming.AnalyticsSender>`.


Viewing the TensorBoard Dashboard during Training
--------------------------------------------------

Now you can use admin commands to upload, deploy, and start this example app. To do this on a proof of concept local
FL system, follow the sections :ref:`setting_up_poc` and :ref:`starting_poc` if you have not already.

Log into the Admin client by entering ``admin`` for both the username and password.
Then, use these Admin commands to run the experiment:

.. code-block:: shell

    > set_run_number 1
    > upload_app hello-pt-tb
    > deploy_app hello-pt-tb
    > start_app all


On the client side, the ``AnalyticsSender`` works as a TensorBoard SummaryWriter. Instead of writing to TB files, it actually generates NVFLARE events of type ``analytix_log_stats``.
The ``ConvertToFedEvent`` widget will turn the event ``analytix_log_stats`` into a fed event ``fed.analytix_log_stats``, which will be delivered to the server side.

On the server side, the ``TBAnalyticsReceiver`` is configured to process ``fed.analytix_log_stats`` events, which writes received TB data into appropriate TB files on the server
(defaults to ``server/run_1/tb_events``).

To view training metrics that are being streamed to the server, run:

.. code-block:: shell

   tensorboard --logdir=poc/server/run_1/tb_events

Note: if the server is running on a remote machine, use port forwarding to view the TensorBoard dashboard in a browser. For example:

.. code-block:: shell

   ssh -L {local_machine_port}:127.0.0.1:6006 user@server_ip

Congratulations! Now you will be able to see the live training metrics of each client from a central place on the server.