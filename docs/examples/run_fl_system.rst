Now you can use admin command prompt to submit and start this example job.
To do this on a proof of concept local FL system, follow the sections
:ref:`setting_up_poc` and :ref:`starting_poc` if you have not already.

Running the FL System
^^^^^^^^^^^^^^^^^^^^^

With the admin client command prompt successfully connected and logged in, enter the command below.

.. parsed-literal::

    > submit_job |ExampleApp|

Pay close attention to what happens in each of four terminals.
You can see how the admin submits the job to the server and how
the :class:`JobRunner <nvflare.private.fed.server.job_runner.JobRunner>` on the server
automatically picks up the job to deploy and start the run.

This command uploads the job configuration from the admin client to the server.
A job id will be returned, and we can use that id to access job information.

.. note::

    If we use submit_job [app] then that app will be treated as a single app job.

From time to time, you can issue ``check_status server`` in the admin client to check the entire training progress.

You should now see how the training does in the very first terminal (the one that started the server).
