.. _tensorboard_streaming:

FL Experiment Tracking with TensorBoard Streaming
=================================================

Introduction
-------------

In this exercise, you will learn how to stream TensorBoard events from the clients
to the server in order to visualize live training metrics from a central place on the server.

This exercise will be working with the ``tensorboard`` example in the advanced examples folder under experiment-tracking,
which builds upon :doc:`hello_pt_job_api` by adding TensorBoard streaming.

The setup of this exercise consists of one **server** and two **clients**.

.. note::

  This exercise differs from :doc:`hello_pt_job_api`, as it uses the ``Learner`` API along with the ``LearnerExecutor``.
  In short, the execution flow is abstracted away into the ``LearnerExecutor``, allowing you to only need to implement the required methods in the ``Learner`` class.
  This will not be the focus of this guide, however you can learn more at :class:`Learner<nvflare.app_common.abstract.learner_spec.Learner>`
  and :class:`LearnerExecutor<nvflare.app_common.executors.learner_executor.LearnerExecutor>`.


Let's get started. Make sure you have an environment with NVIDIA FLARE installed as described in
:ref:`getting_started`. First clone the repo:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Now remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.
And install the required dependencies in the example folder (NVFlare/examples/advanced/experiment-tracking/tensorboard).

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install -r requirements.txt


Adding TensorBoard Streaming to Configurations
------------------------------------------------

Inside the example, job configuration and TensorBoard setup are defined in:

- :github_nvflare_link:`job.py <examples/advanced/experiment-tracking/tensorboard/job.py>`
- :github_nvflare_link:`client.py <examples/advanced/experiment-tracking/tensorboard/client.py>`

Take a look at the components section of the client config at line 24.
The first component is the ``pt_learner`` which contains the initialization, training, and validation logic.
``learner_with_tb.py`` (under NVFlare/examples/advanced/experiment-tracking/pt) is where we will add our TensorBoard streaming changes.

Next we have the :class:`TBWriter<nvflare.app_opt.tracking.tb.tb_writer.TBWriter>`,
which implements some common methods that follow the signatures from the PyTorch SummaryWriter.
This makes it easy for the ``pt_learner`` to log metrics and send events.

Finally, we have the :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>`,
which converts local events to federated events.
This changes the event ``analytix_log_stats`` into a fed event ``fed.analytix_log_stats``,
which will then be streamed from the clients to the server.

Under the component section in the server config, we have the
:class:`TBAnalyticsReceiver<nvflare.app_common.pt.tb_receiver.TBAnalyticsReceiver>`
of type :class:`AnalyticsReceiver<nvflare.app_common.widgets.streaming.AnalyticsReceiver>`.

This component receives TensorBoard events from the clients and saves them to a specified folder
(default ``tb_events``) under the server's run folder.

Notice how the accepted event type ``"fed.analytix_log_stats"`` matches the output of
:class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` in the client config.


Adding TensorBoard Streaming to your Code
-------------------------------------------

In this exercise, TensorBoard logging is implemented in:

- :github_nvflare_link:`client.py <examples/advanced/experiment-tracking/tensorboard/client.py>`

First we must initialize our TensorBoard writer to the ``AnalyticsSender`` we defined in the client config:

The ``LearnerExecutor`` passes in the component dictionary into the ``parts`` parameter of ``initialize()``.
We can then access the ``AnalyticsSender`` component we defined in ``config_fed_client.json``
by using the ``self.analytic_sender_id`` as the key in the ``parts`` dictionary.
Note that ``self.analytic_sender_id`` defaults to ``"analytic_sender"``,
but we can also define it in the client config to be passed into the constructor.

Now that our TensorBoard writer is set to ``AnalyticsSender``,
we can write and stream training metrics to the server in ``local_train()``:

The script uses ``add_scalar(tag, scalar, global_step)`` to send training metrics and validation accuracy.

You can learn more about other supported writer methods in
:class:`AnalyticsSender<nvflare.app_common.widgets.streaming.AnalyticsSender>`.


Train the Model, Federated!
---------------------------

.. |ExampleApp| replace:: tensorboard-streaming
.. include:: run_fl_system.rst


Viewing the TensorBoard Dashboard during Training
--------------------------------------------------

On the client side, the ``AnalyticsSender`` works as a TensorBoard SummaryWriter.
Instead of writing to TB files, it actually generates NVFLARE events of type ``analytix_log_stats``.

The ``ConvertToFedEvent`` widget will turn the event ``analytix_log_stats`` into a fed event
``fed.analytix_log_stats``, which will be delivered to the server side.

On the server side, the ``TBAnalyticsReceiver`` is configured to process ``fed.analytix_log_stats`` events,
which writes received TB data into appropriate TB files on the server
(defaults to ``server/[JOB ID]/tb_events``).

To view training metrics that are being streamed to the server, run:

.. code-block:: shell

   tensorboard --logdir=poc/server/[JOB ID]/tb_events

.. note::

    if the server is running on a remote machine, use port forwarding to view the TensorBoard dashboard in a browser.
    For example:

    .. code-block:: shell

       ssh -L {local_machine_port}:127.0.0.1:6006 user@server_ip

.. attention::

   The ``server/[JOB ID]`` folder only exists when job is running.
   After the job is finished, please use `download_job [JOB ID]` to get the workspace data as explained below.

.. include:: access_result.rst

.. include:: shutdown_fl_system.rst

Congratulations!

Now you will be able to see the live training metrics of each client from a central place on the server.

The full source code for this exercise can be found in
:github_nvflare_link:`examples/advanced/experiment-tracking/tensorboard <examples/advanced/experiment-tracking/tensorboard>`.

Previous Versions of TensorBoard Streaming
------------------------------------------

   - `tensorboard-streaming for 2.3 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/advanced/experiment-tracking/tensorboard-streaming>`_
