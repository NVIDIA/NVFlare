######################
NVIDIA FLARE Workspace
######################

NVIDIA FLARE maintains a workspace for keeping the FL apps and execution results for different jobs
under folders with the name of the ``job_id``.

The following is the workspace folder structure when running NVIDIA FLARE for the server and clients.

.. _server_workspace:

******
Server
******

.. code-block:: shell

    /some_path_on_fl_server/fl_server_workspace_root/
        admin_audit.log
        log.txt
        startup/
            authorization.json
            fed_server.json
            log.config
            readme.txt
            rootCA.pem
            server_context.tenseal
            server.crt
            server.key
            signature.pkl
            start.sh
            stop_fl.sh
            sub_start.sh
        transfer/
        aefdb0a3-6fbb-4c53-a677-b6951d6845a6/
            app_server/
                ...
                config_fed_server.json
            fl_app.txt
            log.txt
        baaf8789-e83f-4863-b085-3ca95303e6bc/
            app_server/
                ...
                config/
                    config_fed_server.json
            fl_app.txt
            log.txt

In each ``job_id`` folder, there is the ``app_server`` folder that contains the :ref:`application` that is running
on the server for this ``job_id``.

The ``log.txt`` inside each ``job_id`` folder are the loggings of this job.

While the ``log.txt`` under server folder is the log for the server control process.

The ``startup`` folder contains the config and the scripts to start the FL server program.

.. _access_server_workspace:

Accessing server-side workspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the job is running, each job will have a corresponding workspace under the ``server`` folder.

When the job is finished, the server side workspace will be removed.
The workspace will be saved into the JobStorage.

You can issue the ``download_job [JOB_ID]`` in the admin client to download the server side workspace.

The downloaded workspace will be in ``[DOWNLOAD_DIR]/[JOB_ID]/workspace/``.

.. note::

    If you issue ``download_job`` before the job is finished, the workspace folder will be empty.


.. _client_workspace:

******
Client
******

.. code-block:: shell

    /some_path_on_fl_client/fl_client_workspace_root/
        log.txt
        startup/
            client_context.tenseal
            client.crt
            client.key
            fed_client.json
            log.config
            readme.txt
            rootCA.pem
            signature.pkl
            start.sh
            stop_fl.sh
            sub_start.sh
        transfer/
        aefdb0a3-6fbb-4c53-a677-b6951d6845a6/
            app_clientA/
                ...
                config_fed_client.json
            fl_app.txt
            log.txt
        baaf8789-e83f-4863-b085-3ca95303e6bc/
            app_clientA/
                ...
                config/
                    config_fed_client.json
            fl_app.txt
            log.txt

In each ``job_id`` folder, there is the ``app_clientname`` folder that contains the :ref:`application` that is running
on the client for this ``job_id``.

The ``log.txt`` inside each ``job_id`` folder are the loggings of this job.

While the ``log.txt`` under client folder is the log for the client control process.

The ``startup`` folder contains the config and the scripts to start the FL client program.

The :class:`Workspace<nvflare.apis.workspace.Workspace>` object is available through the FLContext.
From the Workspace, you can access each folder location accordingly

.. code-block:: python

    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)


.. literalinclude:: ../../nvflare/apis/workspace.py
    :language: python
    :lines: 36-
