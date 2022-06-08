######################
NVIDIA FLARE Workspace
######################

NVIDIA FLARE maintains a workspace for keeping the FL apps and execution results for different jobs under folders with
the name of the ``job_id``.  The following is the workspace folder structure when running NVIDIA FLARE for the server and clients.

******
Server
******
::

    server/
        aefdb0a3-6fbb-4c53-a677-b6951d6845a6/
            app_server/
                ...
                config_fed_server.json
            fl_app.txt
        baaf8789-e83f-4863-b085-3ca95303e6bc/
            app_server/
                ...
                config/
                    config_fed_server.json
            fl_app.txt
        startup/
            fed_server.json
            log.config
            start.sh
            sub_start.sh


******
Client
******
::

    clientA/
        aefdb0a3-6fbb-4c53-a677-b6951d6845a6/
            app_clientA/
                ...
                config_fed_client.json
            fl_app.txt
        baaf8789-e83f-4863-b085-3ca95303e6bc/
            app_clientA/
                ...
                config/
                    config_fed_client.json
            fl_app.txt
        startup/
            fed_client.json
            log.config
            start.sh
            sub_start.sh

The FL application and run data from different ``job_ids`` are kept in the separate folders for each job. Under the job
folder, the ``app_server`` folder holds the FL application content of the server. The
``app_clientName`` folder holds the FL application content of the client. The config_fed_server.json and
config_fed_client.json may be in the root folder of the FL App, or in a sub-folder (for example: config) of the FL App.

The ``startup`` folder contains the config and the scripts to start the FL server and client program.

The :class:`Workspace<nvflare.apis.workspace.Workspace>` object is available through the FLContext. From the Workspace,
you can access each folder location accordingly::

    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)


.. literalinclude:: ../../nvflare/apis/workspace.py
    :language: python
    :lines: 36-
