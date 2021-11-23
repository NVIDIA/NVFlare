.. _FAQ:

###
FAQ
###

************************
Client related questions
************************

#. What happens if an FL client joins during the FL training?

    An FL client can join the FL training any time. As long as the participating FL clients are still within the maximum
    number of clients, the client can join. The newly joined client will get the current round of global model for training
    and will contribute to the current global model.

#. Do federated learning clients need to open any ports for the FL server to reach the FL client?

    No, federated learning training does not require for FL clients to open their network for inbound traffic. The server
    never sends uninvited requests to clients but only responds to client requests.

.. _multi gpu training:

#. Can a client train with multiple GPUs?

    Yes, the administrator command ``start_mgpu client <gpu number> <client name>`` can be used to start training
    on FL clients with different numbers of GPUs when starting training. But you will need to install ``torch`` for
    PyTorch based Trainer, and ``mpi4py`` for all other trainers. It is possible for different clients to be
    training with different numbers of GPUs.

#. How do FL clients get identified?

    The federated learning clients are identified by a dynamically generated FL token issued by the server during runtime.
    When an FL client first joins an FL training, it first needs to send a login request to the FL server. During the login
    process, the FL server and client need to exchange SSL certificates for bi-directional authentication. Once the
    authentication is successful, the FL server sends an FL token to the client. The FL client will use this FL token to
    identify itself for all following requests for the global model and all model updating operations.

#. Can I run multiple FL clients from the same machine?

    Yes. The FL clients are identified by FL token, not machine IP. Each FL client will have its own FL token as well as
    instance name, which is the client name that must be used for issuing specific commands to that client.

#. Can I use the same client package to run multiple instances for the same client?

    Yes, you can start multiple instances of FL clients from the same client packages. Each FL client will be identified
    by its unique instance names, for example: "flclient1", "flclient1_1", "flclient1_2", etc. The instance name must be
    used for issuing specific commands to that client from the admin tool.

#. What happens if a federated learning client crashes?

    Federated learning clients will send a heartbeat call to the FL server once every minute. If an FL client crashes and
    the FL server does not get a heartbeat from that client for 10 minutes (can be set with "heart_beat_timeout" in the
    server's config json), the FL server will remove that client from the training client list.

#. Can FL clients join or quit in the middle of federated learning training?

    Yes, an FL client can join or quit in the middle of the FL training at any time. The client will pick up the global
    model at the current round of the server to participate in the FL training. When quitting, the FL server will
    automatically remove the FL client after it quits and no heartbeat is received for the duration of the
    "heart_beat_timeout" configured on the server. If using an admin tool, it is recommended to use the "abort" and
    "shutdown" commands to gracefully stop the clients.

#. What if the number of participating FL clients is below the minimum number of clients required?

    When an FL client passes authentication, it can request the current round of the global model and starts the FL training right away.
    There is no need to wait for other clients. Once the client finishes its own training, it will send the update to the server
    for aggregation. However, if the server does not receive enough updates from other clients, the FL server will not start
    the next round of FL training. The finished FL client will be waiting for the next round's model.

#. What happens if more than the minimum numbers of FL clients submit an updated model?

    The FL server begins model aggregation after accepting updates from the minimum number of FL clients required and
    waiting for "wait_after_min_clients" configured on the server. The updates that are received after this will be
    discarded. All the clients will get the next round of the global model to start the next round FL training.

#. How does a client decide to quit federated learning training?

    The FL client always asks the server for the current round of training. If the server is not ready, the FL client will wait.
    The client will only stop if the server becomes unreachable. The FL client can also be killed with the admin tool
    issuing a "shutdown" command or by ctrl-C on the client itself, although this is not recommended because the server
    will wait for the duration of "heart_beat_timeout" before knowing that the client has stopped.

************************
Server related questions
************************

#. What happens if the FL server crashes?

    There are two scenarios for the FL server crashing during the FL training. If the server crashes when the FL client is trying to
    connect to the server for model exchange, the FL client will continue to attempt connecting to the server for up to 30 seconds.
    If the server is still down after that, FL client will shut itself down. If the server crashes during the FL client model training,
    as long as the server restarts before the FL client attempts model updating, it will have no impact to the FL clients.

    When restarting the FL server, you can find the previous training round number from the previous log. Then you can choose to
    train from scratch or continuously using previous training model.

#. Does the federated learning server need a GPU?

    No, there is no need to have GPU on the server side for the FL server to deploy. However, certain handlers may require
    GPUs. To disable GPUs on the server, include the following in the shell script that runs the server::

        export CUDA_VISIBLE_DEVICES=

#. What port do I need to open from the firewall on the FL server network?

    Depending on the configuration of :ref:`project.yaml <project_yml>` which controls which port the gRPC is deployed to,
    the FL server network needs to open that port for outside clients to reach the FL server.

#. What if the federated learning server is behind a load balancer?

    Currently, federated learning does not support load balancing between multiple FL servers.

***************************************
Overall training flow related questions
***************************************

#. How does the federated learning server decide when to stop FL?

    The FL server always runs from the "start_round" to "num_rounds". The FL server will stop the training when the
    current round meets "num_rounds".

#. Can I run the FL server on AWS while running the FL client within my institution?

    Yes, use the AWS instance name as the server cn in project.yml file. (e.g.: ec2-3-99-123-456.compute-1.amazonaws.com)

#. How can I deploy different applications for different clients?

    You can edit the application folder for each individual client on your desktop, then upload and deploy to each individual client
    with the admin tool. Each client can run with its own application configuration.

#. Can I use the same "run_number" as previously used?

    Yes, you can re-use the same "run_number" as previously used. The "run_number" serves as an FL training workspace. The
    FL training logs, such as tensorboard, training stats, etc, are stored within the same "run_number" workspace.

#. What should I do if the admin notices one client's training is behaving erroneously or unexpectedly?

    The admin can issue a command to abort the FL client training: ``abort client client_name``. If the command is issued
    without the client_name, then the command will be sent to all the clients. Because of the nature of model training, it
    may take a little time for the FL client to completely stop. Use the "check_status client client_name" command to see
    if the client status is "stopped".

#. Why do the admin commands to the clients have a long delay before getting a response?

    The admin commands to the clients pass through the server. If for some reason the command is delayed by the network, or
    if the client command takes a long time to process, the admin console will experience a delay for the response. The
    default timeout is 10 seconds. You can use the “set_timeout” command to adjust the command timeout. If this timeout
    value is set too low, the admin command may not reach the client to execute the command.

#. Once the FL training has started, can I use the same server / client set up to train different models?

    Yes, you can upload different applications to the server and clients to train different models. Make sure to use the
    "run_number" to keep your trained models in different run spaces without confusing the models. The FL system only
    completes when the admin issues the “shutdown” command or ctrl-C is used to end the process.

#. Why if my custom components not updating between runs?

    If you want to change the code you have already loaded in a custom component, it is recommended that you add a
    version number or change the class name slightly. Python does not load new code definitions with the same class name
    and by default Python does not allow the loaded modules to be removed. With a version or altered name, Python will
    be able to treat the code as new and load it from the sys.path.

#. Why do commands sometimes fail?

    Sometimes if you are trying to check status of the client and the server is already busy transferring the model and
    does not have extra bandwidth for the command, the command may time out. In that case, please wait and try again.

*********************
Cross Site Validation
*********************

#. I don't want to share my local model. Can I opt out of cross site validation?

    Cross site validation is opt-out be default. To opt in, you must set "cross_site_validate" to true in config_fed_client.json.

#. Cross site validation has finished. How can I see the results? 

    Use the admin commands "validate all" or "validate <client_1> <client2>" to retrieve the results.

#. Cross site validation ran but my results are empty. Why?

    If some client is not participating in cross_site_validation OR an error occurs during validation, you will see empty results 
    for that section. Please use the logs to retrieve the error.

#. My client is stuck in endless loop of asking for models, then waiting and repeat. What do I do?

    In some cases, cross site validation may get stuck. This is because the server sometimes doesn't know when (or if ever) a model 
    will become available. In these cases, please use the "abort <client>" to stop cross site validation manually.

#. I called "abort client" during training and it started cross site validation. Why?

    This is the intended behavior. If a client is aborted, it will transition to cross site validation phase (if participating). To
    completely abort, call "abort client" command again.

************
Known issues
************

#. If server dies and then is restarted, intentionally or unintentionally, all clients will have to be restarted.
#. Running out of memory can happen at any time, especially if the server and clients are running on same machine.
   This can cause the server to die unexpectedly.
#. Putting applications in the transfer folders without using the upload_app command or forgetting to delete the models
   folder inside, a mysterious error may occur when running the deploy_app command because the application folder is too
   large to be uploaded and that causes timeout.
#. Please don't start a new training run or start a new app before the previous application is fully stopped. Or users
   can do ``abort client`` and ``abort server`` before ``start_app`` for the new run.
