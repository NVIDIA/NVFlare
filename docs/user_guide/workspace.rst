######################
NVIDIA FLARE Workspace
######################

NVIDIA FLARE maintains a workspace for keeping the FL apps and execution results of different ``run_number``
experiments. The following is the workspace folder structure when running NVIDIA FLARE for the server and clients.

******
Server
******
::

    server/
        run_1/
            app_server/
                ...
                config_fed_server.json
            fl_app.txt
        run_2/
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
        run_1/
            app_clientA/
                ...
                config_fed_client.json
            fl_app.txt
        run_2/
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

The FL application and run data from different ``run_number`` are kept in the separate ``run_N`` folders for each run
number ``N``. Under the ``run_N`` folder, the ``app_server`` folder holds the FL application content of the server. The
``app_clientName`` folder holds the FL application content of the client. The config_fed_server.json and
config_fed_client.json may be in the root folder of the FL App, or in a sub-folder (for example: config) of the FL App.

The fl_app.txt indicates which FL App current ``run_number`` is using. The ``deploy_app`` command will deploy the
corresponding FL app into the current ``run_number``. If the ``app_server`` or ``app_clientName`` folder
already exists, all the contents within that folder will be wiped out and deployed a new FL application. If you would
like to keep some execution results within the same ``run_number``, you can keep them in the root directory of the
``run_N`` folder.

The ``startup`` folder contains the config and the scripts to start the FL server and client program.

The :class:`Workspace<nvflare.apis.workspace.Workspace>` object is available through the FLContext. From the Workspace,
you can access each folder location accordingly::

    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)


.. literalinclude:: ../../nvflare/apis/workspace.py
    :language: python
    :lines: 36-
