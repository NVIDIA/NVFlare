.. _fl_simulator:

#########################
NVIDIA FLARE FL Simulator
#########################

The NVIDIA FLARE FL Simulator can help researchers
accelerate the development of federated learning workflows.

The FL Simulator is a lightweight simulator of a running NVFLARE FL deployment,
and it can allow researchers to test and debug their application without
provisioning a real project.

The FL jobs run on a server and 
multiple clients in the same process but in a similar way to how it would run
in a real deployment so researchers can more quickly build out new components
and jobs that can then be directly used in a real production deployment.

***********************
Command Usage
***********************

.. code-block:: shell

    $ nvflare simulator -h
    usage: nvflare simulator [-h] [-w WORKSPACE] [-n N_CLIENTS] [-c CLIENTS] [-t THREADS] [-gpu GPU] [-m MAX_CLIENTS] [--end_run_for_all] job_folder

    positional arguments:
        job_folder

    options:
        -h, --help            show this help message and exit
        -w WORKSPACE, --workspace WORKSPACE
                            WORKSPACE folder
        -n N_CLIENTS, --n_clients N_CLIENTS
                            number of clients
        -c CLIENTS, --clients CLIENTS
                            client names list
        -t THREADS, --threads THREADS
                            number of parallel running clients
        -gpu GPU, --gpu GPU   list of GPU Device Ids, comma separated
        -m MAX_CLIENTS, --max_clients MAX_CLIENTS
                            max number of clients
        --end_run_for_all     flag to indicate if running END_RUN event for all clients

    
*****************
Command examples
*****************

Run a single NVFlare app
========================

This command will run the same ``hello-numpy-sag`` app on the server and 8 clients using 1 process. The client names will be site-1, site-2, ... , site-8:

.. code-block:: python

    nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -n 8 -t 1

.. raw:: html

   <details>
   <summary><a>Example Output</a></summary>

.. code-block:: none

    2022-10-17 17:30:28,165 - SimulatorRunner - INFO - Create the Simulator Server.
    2022-10-17 17:30:28,225 - nvflare.fuel.hci.server.hci - INFO - Starting Admin Server localhost on Port 57293
    2022-10-17 17:30:28,227 - SimulatorServer - INFO - starting insecure server at localhost:42815
    2022-10-17 17:30:28,229 - SimulatorRunner - INFO - Deploy the Apps.
    2022-10-17 17:30:28,231 - SimulatorRunner - INFO - Create the simulate clients.
    2022-10-17 17:30:28,306 - ClientManager - INFO - Client: New client site-1@127.0.0.1 joined. Sent token: 529ce6b4-5d71-4fe5-b6fc-ed9d14d26936.  Total clients: 1
    2022-10-17 17:30:28,306 - FederatedClient - INFO - Successfully registered client:site-1 for project simulator_server. Token:529ce6b4-5d71-4fe5-b6fc-ed9d14d26936 SSID:
    2022-10-17 17:30:28,381 - ClientManager - INFO - Client: New client site-2@127.0.0.1 joined. Sent token: 3d9420db-1aa0-4142-adbb-2d8fc87a8e8b.  Total clients: 2
    2022-10-17 17:30:28,381 - FederatedClient - INFO - Successfully registered client:site-2 for project simulator_server. Token:3d9420db-1aa0-4142-adbb-2d8fc87a8e8b SSID:
    2022-10-17 17:30:28,448 - ClientManager - INFO - Client: New client site-3@127.0.0.1 joined. Sent token: 738e9f46-877c-4856-bbd2-674eea8f5f27.  Total clients: 3
    2022-10-17 17:30:28,448 - FederatedClient - INFO - Successfully registered client:site-3 for project simulator_server. Token:738e9f46-877c-4856-bbd2-674eea8f5f27 SSID:
    2022-10-17 17:30:28,524 - ClientManager - INFO - Client: New client site-4@127.0.0.1 joined. Sent token: 2e9e56a9-ad05-48d0-bc60-9d322865f33e.  Total clients: 4
    2022-10-17 17:30:28,524 - FederatedClient - INFO - Successfully registered client:site-4 for project simulator_server. Token:2e9e56a9-ad05-48d0-bc60-9d322865f33e SSID:
    2022-10-17 17:30:28,591 - ClientManager - INFO - Client: New client site-5@127.0.0.1 joined. Sent token: 7e822d77-7a7b-4ea4-9e67-b971416e456e.  Total clients: 5
    2022-10-17 17:30:28,591 - FederatedClient - INFO - Successfully registered client:site-5 for project simulator_server. Token:7e822d77-7a7b-4ea4-9e67-b971416e456e SSID:
    2022-10-17 17:30:28,666 - ClientManager - INFO - Client: New client site-6@127.0.0.1 joined. Sent token: 0b291c05-0495-4936-aba8-69e735f03528.  Total clients: 6
    2022-10-17 17:30:28,667 - FederatedClient - INFO - Successfully registered client:site-6 for project simulator_server. Token:0b291c05-0495-4936-aba8-69e735f03528 SSID:
    2022-10-17 17:30:28,734 - ClientManager - INFO - Client: New client site-7@127.0.0.1 joined. Sent token: e6127906-5283-45e3-b510-2866ff8a51a4.  Total clients: 7
    2022-10-17 17:30:28,734 - FederatedClient - INFO - Successfully registered client:site-7 for project simulator_server. Token:e6127906-5283-45e3-b510-2866ff8a51a4 SSID:
    2022-10-17 17:30:28,801 - ClientManager - INFO - Client: New client site-8@127.0.0.1 joined. Sent token: 8fe14500-a6b8-47ad-b50c-aa22f9827830.  Total clients: 8
    2022-10-17 17:30:28,801 - FederatedClient - INFO - Successfully registered client:site-8 for project simulator_server. Token:8fe14500-a6b8-47ad-b50c-aa22f9827830 SSID:
    2022-10-17 17:30:28,802 - SimulatorRunner - INFO - Set the client status ready.
    2022-10-17 17:30:28,803 - SimulatorRunner - INFO - Deploy and start the Server App.
    2022-10-17 17:30:29,254 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job]: Server runner starting ...
    2022-10-17 17:30:29,254 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job]: starting workflow scatter_and_gather (<class 'nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather'>) ...
    2022-10-17 17:30:29,254 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job]: Initializing ScatterAndGather workflow.
    2022-10-17 17:30:29,255 - NPModelPersistor - INFO - [identity=simulator_server, run=simulate_job]: Unable to load model from /tmp/nvflare/workspace_folder/simulate_job/models/server.npy: [Errno 2] No such file or directory: '/tmp/nvflare/workspace_folder/simulate_job/models/server.npy'. Using default data instead.
    2022-10-17 17:30:29,255 - NPModelPersistor - INFO - [identity=simulator_server, run=simulate_job]: Loaded initial model: {'numpy_key': array([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]], dtype=float32)}
    2022-10-17 17:30:29,256 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job]: Workflow scatter_and_gather (<class 'nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather'>) started
    2022-10-17 17:30:29,256 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Beginning ScatterAndGather training phase.
    2022-10-17 17:30:29,256 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Round 0 started.
    2022-10-17 17:30:29,256 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: scheduled task train
    2022-10-17 17:30:29,804 - SimulatorClientRunner - INFO - Start the clients run simulation.
    2022-10-17 17:30:30,806 - SimulatorClientRunner - INFO - Simulate Run client: site-1
    E1017 17:30:30.807042185   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:32,301 - ClientRunner - INFO - [identity=site-1, run=simulate_job]: client runner started
    2022-10-17 17:30:32,301 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-1
    2022-10-17 17:30:32,375 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: assigned task to client site-1: name=train, id=5f504355-2edf-4f6a-9cc5-56181f95f28d
    2022-10-17 17:30:32,375 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: sent task assignment to client
    2022-10-17 17:30:32,376 - SimulatorServer - INFO - GetTask: Return task: train to client: site-1 (529ce6b4-5d71-4fe5-b6fc-ed9d14d26936) 
    2022-10-17 17:30:32,376 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.07400965690612793 seconds
    2022-10-17 17:30:32,377 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:32,378 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=5f504355-2edf-4f6a-9cc5-56181f95f28d
    2022-10-17 17:30:32,378 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:32,379 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Task name: train
    2022-10-17 17:30:32,379 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:32,379 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Model: 
    {'numpy_key': array([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]], dtype=float32)}
    2022-10-17 17:30:32,379 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Current Round: 0
    2022-10-17 17:30:32,379 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Total Rounds: 3
    2022-10-17 17:30:32,379 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Client identity: site-1
    2022-10-17 17:30:32,380 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:32,380 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Model after training: {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:32,380 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: finished processing task
    2022-10-17 17:30:32,381 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:32,382 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:32,459 - SimulatorServer - INFO - received update from simulator_server_site-1_0 (1140 Bytes, 1666042232 seconds)
    2022-10-17 17:30:32,459 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job]: got result from client site-1 for task: name=train, id=5f504355-2edf-4f6a-9cc5-56181f95f28d
    2022-10-17 17:30:32,460 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: invoking result_received_cb ...
    2022-10-17 17:30:32,460 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-1 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:32,460 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Aggregation_weight missing for site-1 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:32,460 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: Contribution from site-1 ACCEPTED by the aggregator.
    2022-10-17 17:30:32,460 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:32,461 - Communicator - INFO - Received comments: simulator_server Received from site-1 (1140 Bytes, 1666042232 seconds). SubmitUpdate time: 0.07916855812072754 seconds
    2022-10-17 17:30:32,462 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5f504355-2edf-4f6a-9cc5-56181f95f28d]: result sent to server for task: name=train, id=5f504355-2edf-4f6a-9cc5-56181f95f28d
    2022-10-17 17:30:32,462 - ClientTaskWorker - INFO - Finished one task run for client: site-1
    2022-10-17 17:30:32,462 - SimulatorClientRunner - INFO - Simulate Run client: site-2
    2022-10-17 17:30:32,462 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-1 
    E1017 17:30:33.464621867   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:34,956 - ClientRunner - INFO - [identity=site-2, run=simulate_job]: client runner started
    2022-10-17 17:30:34,956 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-2
    2022-10-17 17:30:35,045 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: assigned task to client site-2: name=train, id=3a5cf17a-515e-48a6-87fc-f920b03221e1
    2022-10-17 17:30:35,046 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: sent task assignment to client
    2022-10-17 17:30:35,046 - SimulatorServer - INFO - GetTask: Return task: train to client: site-2 (3d9420db-1aa0-4142-adbb-2d8fc87a8e8b) 
    2022-10-17 17:30:35,047 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.08929610252380371 seconds
    2022-10-17 17:30:35,048 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:35,049 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=3a5cf17a-515e-48a6-87fc-f920b03221e1
    2022-10-17 17:30:35,049 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:35,049 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Task name: train
    2022-10-17 17:30:35,049 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:35,050 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Model: 
    {'numpy_key': array([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]], dtype=float32)}
    2022-10-17 17:30:35,050 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Current Round: 0
    2022-10-17 17:30:35,050 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Total Rounds: 3
    2022-10-17 17:30:35,050 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Client identity: site-2
    2022-10-17 17:30:35,050 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:35,050 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Model after training: {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:35,051 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: finished processing task
    2022-10-17 17:30:35,052 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:35,052 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:35,121 - SimulatorServer - INFO - received update from simulator_server_site-2_0 (1140 Bytes, 1666042235 seconds)
    2022-10-17 17:30:35,121 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job]: got result from client site-2 for task: name=train, id=3a5cf17a-515e-48a6-87fc-f920b03221e1
    2022-10-17 17:30:35,122 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: invoking result_received_cb ...
    2022-10-17 17:30:35,122 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-2 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:35,122 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Aggregation_weight missing for site-2 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:35,123 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: Contribution from site-2 ACCEPTED by the aggregator.
    2022-10-17 17:30:35,123 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:35,123 - Communicator - INFO - Received comments: simulator_server Received from site-2 (1140 Bytes, 1666042235 seconds). SubmitUpdate time: 0.07082653045654297 seconds
    2022-10-17 17:30:35,124 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3a5cf17a-515e-48a6-87fc-f920b03221e1]: result sent to server for task: name=train, id=3a5cf17a-515e-48a6-87fc-f920b03221e1
    2022-10-17 17:30:35,124 - ClientTaskWorker - INFO - Finished one task run for client: site-2
    2022-10-17 17:30:35,124 - SimulatorClientRunner - INFO - Simulate Run client: site-3
    2022-10-17 17:30:35,125 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-2 
    2022-10-17 17:30:37,631 - ClientRunner - INFO - [identity=site-3, run=simulate_job]: client runner started
    2022-10-17 17:30:37,631 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-3
    2022-10-17 17:30:37,704 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: assigned task to client site-3: name=train, id=5b985bfb-6a58-437f-b984-720455c1e20b
    2022-10-17 17:30:37,704 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: sent task assignment to client
    2022-10-17 17:30:37,705 - SimulatorServer - INFO - GetTask: Return task: train to client: site-3 (738e9f46-877c-4856-bbd2-674eea8f5f27) 
    2022-10-17 17:30:37,705 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.0725107192993164 seconds
    2022-10-17 17:30:37,706 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:37,707 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=5b985bfb-6a58-437f-b984-720455c1e20b
    2022-10-17 17:30:37,707 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:37,707 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Task name: train
    2022-10-17 17:30:37,707 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:37,708 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Model: 
    {'numpy_key': array([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]], dtype=float32)}
    2022-10-17 17:30:37,708 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Current Round: 0
    2022-10-17 17:30:37,708 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Total Rounds: 3
    2022-10-17 17:30:37,708 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Client identity: site-3
    2022-10-17 17:30:37,708 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:37,709 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Model after training: {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:37,709 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: finished processing task
    2022-10-17 17:30:37,710 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:37,711 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:37,779 - SimulatorServer - INFO - received update from simulator_server_site-3_0 (1140 Bytes, 1666042237 seconds)
    2022-10-17 17:30:37,780 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job]: got result from client site-3 for task: name=train, id=5b985bfb-6a58-437f-b984-720455c1e20b
    2022-10-17 17:30:37,780 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: invoking result_received_cb ...
    2022-10-17 17:30:37,780 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-3 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:37,780 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Aggregation_weight missing for site-3 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:37,781 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: Contribution from site-3 ACCEPTED by the aggregator.
    2022-10-17 17:30:37,781 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:37,781 - Communicator - INFO - Received comments: simulator_server Received from site-3 (1140 Bytes, 1666042237 seconds). SubmitUpdate time: 0.07065510749816895 seconds
    2022-10-17 17:30:37,782 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=5b985bfb-6a58-437f-b984-720455c1e20b]: result sent to server for task: name=train, id=5b985bfb-6a58-437f-b984-720455c1e20b
    2022-10-17 17:30:37,782 - ClientTaskWorker - INFO - Finished one task run for client: site-3
    2022-10-17 17:30:37,783 - SimulatorClientRunner - INFO - Simulate Run client: site-4
    2022-10-17 17:30:37,783 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-3 
    E1017 17:30:38.785133258   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:40,280 - ClientRunner - INFO - [identity=site-4, run=simulate_job]: client runner started
    2022-10-17 17:30:40,280 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-4
    2022-10-17 17:30:40,351 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: assigned task to client site-4: name=train, id=8b798d8c-4157-4ee9-b207-3f532784154a
    2022-10-17 17:30:40,351 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: sent task assignment to client
    2022-10-17 17:30:40,351 - SimulatorServer - INFO - GetTask: Return task: train to client: site-4 (2e9e56a9-ad05-48d0-bc60-9d322865f33e) 
    2022-10-17 17:30:40,351 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.07045435905456543 seconds
    2022-10-17 17:30:40,353 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:40,353 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=8b798d8c-4157-4ee9-b207-3f532784154a
    2022-10-17 17:30:40,354 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:40,354 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Task name: train
    2022-10-17 17:30:40,354 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:40,354 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Model: 
    {'numpy_key': array([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]], dtype=float32)}
    2022-10-17 17:30:40,354 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Current Round: 0
    2022-10-17 17:30:40,354 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Total Rounds: 3
    2022-10-17 17:30:40,354 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Client identity: site-4
    2022-10-17 17:30:40,355 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:40,355 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Model after training: {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:40,355 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: finished processing task
    2022-10-17 17:30:40,356 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:40,357 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:40,425 - SimulatorServer - INFO - received update from simulator_server_site-4_0 (1140 Bytes, 1666042240 seconds)
    2022-10-17 17:30:40,425 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job]: got result from client site-4 for task: name=train, id=8b798d8c-4157-4ee9-b207-3f532784154a
    2022-10-17 17:30:40,426 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: invoking result_received_cb ...
    2022-10-17 17:30:40,426 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-4 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:40,426 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Aggregation_weight missing for site-4 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:40,426 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: Contribution from site-4 ACCEPTED by the aggregator.
    2022-10-17 17:30:40,426 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:40,427 - Communicator - INFO - Received comments: simulator_server Received from site-4 (1140 Bytes, 1666042240 seconds). SubmitUpdate time: 0.06950092315673828 seconds
    2022-10-17 17:30:40,427 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=8b798d8c-4157-4ee9-b207-3f532784154a]: result sent to server for task: name=train, id=8b798d8c-4157-4ee9-b207-3f532784154a
    2022-10-17 17:30:40,428 - ClientTaskWorker - INFO - Finished one task run for client: site-4
    2022-10-17 17:30:40,428 - SimulatorClientRunner - INFO - Simulate Run client: site-5
    2022-10-17 17:30:40,428 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-4 
    E1017 17:30:41.430357472   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:42,925 - ClientRunner - INFO - [identity=site-5, run=simulate_job]: client runner started
    2022-10-17 17:30:42,925 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-5
    2022-10-17 17:30:43,008 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: assigned task to client site-5: name=train, id=c536b2ff-da85-49ca-a90d-705af6aefbff
    2022-10-17 17:30:43,008 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: sent task assignment to client
    2022-10-17 17:30:43,009 - SimulatorServer - INFO - GetTask: Return task: train to client: site-5 (7e822d77-7a7b-4ea4-9e67-b971416e456e) 
    2022-10-17 17:30:43,009 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.08272647857666016 seconds
    2022-10-17 17:30:43,010 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:43,011 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=c536b2ff-da85-49ca-a90d-705af6aefbff
    2022-10-17 17:30:43,011 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:43,011 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Task name: train
    2022-10-17 17:30:43,011 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:43,012 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Model: 
    {'numpy_key': array([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]], dtype=float32)}
    2022-10-17 17:30:43,012 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Current Round: 0
    2022-10-17 17:30:43,012 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Total Rounds: 3
    2022-10-17 17:30:43,012 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Client identity: site-5
    2022-10-17 17:30:43,012 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:43,013 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Model after training: {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:43,013 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: finished processing task
    2022-10-17 17:30:43,014 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:43,015 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:43,117 - SimulatorServer - INFO - received update from simulator_server_site-5_0 (1140 Bytes, 1666042243 seconds)
    2022-10-17 17:30:43,117 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job]: got result from client site-5 for task: name=train, id=c536b2ff-da85-49ca-a90d-705af6aefbff
    2022-10-17 17:30:43,118 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: invoking result_received_cb ...
    2022-10-17 17:30:43,118 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-5 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:43,118 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Aggregation_weight missing for site-5 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:43,119 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: Contribution from site-5 ACCEPTED by the aggregator.
    2022-10-17 17:30:43,119 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:43,119 - Communicator - INFO - Received comments: simulator_server Received from site-5 (1140 Bytes, 1666042243 seconds). SubmitUpdate time: 0.10463595390319824 seconds
    2022-10-17 17:30:43,120 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c536b2ff-da85-49ca-a90d-705af6aefbff]: result sent to server for task: name=train, id=c536b2ff-da85-49ca-a90d-705af6aefbff
    2022-10-17 17:30:43,120 - ClientTaskWorker - INFO - Finished one task run for client: site-5
    2022-10-17 17:30:43,121 - SimulatorClientRunner - INFO - Simulate Run client: site-6
    2022-10-17 17:30:43,121 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-5 
    2022-10-17 17:30:45,272 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: task train exit with status TaskCompletionStatus.OK
    2022-10-17 17:30:45,641 - ClientRunner - INFO - [identity=site-6, run=simulate_job]: client runner started
    2022-10-17 17:30:45,641 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-6
    2022-10-17 17:30:45,714 - ClientTaskWorker - INFO - Finished one task run for client: site-6
    2022-10-17 17:30:45,714 - SimulatorClientRunner - INFO - Simulate Run client: site-7
    2022-10-17 17:30:45,714 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-6 
    2022-10-17 17:30:45,772 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Start aggregation.
    2022-10-17 17:30:45,773 - DXOAggregator - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: aggregating 5 update(s) at round 0
    2022-10-17 17:30:45,773 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: End aggregation.
    2022-10-17 17:30:45,773 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Start persist model on server.
    2022-10-17 17:30:45,774 - NPModelPersistor - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/models/server.npy
    2022-10-17 17:30:45,774 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: End persist model on server.
    2022-10-17 17:30:45,774 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Round 0 finished.
    2022-10-17 17:30:45,774 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Round 1 started.
    2022-10-17 17:30:45,774 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: scheduled task train
    E1017 17:30:46.716563246   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:48,218 - ClientRunner - INFO - [identity=site-7, run=simulate_job]: client runner started
    2022-10-17 17:30:48,218 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-7
    2022-10-17 17:30:48,291 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: assigned task to client site-7: name=train, id=79dcc239-c29b-4424-99e8-0439d0e1d637
    2022-10-17 17:30:48,292 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: sent task assignment to client
    2022-10-17 17:30:48,292 - SimulatorServer - INFO - GetTask: Return task: train to client: site-7 (e6127906-5283-45e3-b510-2866ff8a51a4) 
    2022-10-17 17:30:48,293 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.07365679740905762 seconds
    2022-10-17 17:30:48,294 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:48,294 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=79dcc239-c29b-4424-99e8-0439d0e1d637
    2022-10-17 17:30:48,295 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:48,295 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Task name: train
    2022-10-17 17:30:48,295 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:48,295 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Model: 
    {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:48,295 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Current Round: 1
    2022-10-17 17:30:48,296 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Total Rounds: 3
    2022-10-17 17:30:48,296 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Client identity: site-7
    2022-10-17 17:30:48,296 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:48,296 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Model after training: {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:30:48,297 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: finished processing task
    2022-10-17 17:30:48,297 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:48,298 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:48,384 - SimulatorServer - INFO - received update from simulator_server_site-7_0 (1140 Bytes, 1666042248 seconds)
    2022-10-17 17:30:48,384 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job]: got result from client site-7 for task: name=train, id=79dcc239-c29b-4424-99e8-0439d0e1d637
    2022-10-17 17:30:48,384 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: invoking result_received_cb ...
    2022-10-17 17:30:48,385 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-7 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:48,385 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Aggregation_weight missing for site-7 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:48,385 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: Contribution from site-7 ACCEPTED by the aggregator.
    2022-10-17 17:30:48,385 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:48,386 - Communicator - INFO - Received comments: simulator_server Received from site-7 (1140 Bytes, 1666042248 seconds). SubmitUpdate time: 0.08739018440246582 seconds
    2022-10-17 17:30:48,387 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=79dcc239-c29b-4424-99e8-0439d0e1d637]: result sent to server for task: name=train, id=79dcc239-c29b-4424-99e8-0439d0e1d637
    2022-10-17 17:30:48,387 - ClientTaskWorker - INFO - Finished one task run for client: site-7
    2022-10-17 17:30:48,387 - SimulatorClientRunner - INFO - Simulate Run client: site-8
    2022-10-17 17:30:48,387 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-7 
    E1017 17:30:49.389404190   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:50,905 - ClientRunner - INFO - [identity=site-8, run=simulate_job]: client runner started
    2022-10-17 17:30:50,905 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-8
    2022-10-17 17:30:50,977 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: assigned task to client site-8: name=train, id=c6207618-9fa1-47b6-8ac0-3375f6139779
    2022-10-17 17:30:50,978 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: sent task assignment to client
    2022-10-17 17:30:50,978 - SimulatorServer - INFO - GetTask: Return task: train to client: site-8 (8fe14500-a6b8-47ad-b50c-aa22f9827830) 
    2022-10-17 17:30:50,979 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.07208514213562012 seconds
    2022-10-17 17:30:50,980 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:50,980 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=c6207618-9fa1-47b6-8ac0-3375f6139779
    2022-10-17 17:30:50,981 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:50,981 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Task name: train
    2022-10-17 17:30:50,981 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:50,981 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Model: 
    {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:50,981 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Current Round: 1
    2022-10-17 17:30:50,981 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Total Rounds: 3
    2022-10-17 17:30:50,981 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Client identity: site-8
    2022-10-17 17:30:50,982 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:50,982 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Model after training: {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:30:50,982 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: finished processing task
    2022-10-17 17:30:50,983 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:50,984 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:51,053 - SimulatorServer - INFO - received update from simulator_server_site-8_0 (1140 Bytes, 1666042251 seconds)
    2022-10-17 17:30:51,053 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job]: got result from client site-8 for task: name=train, id=c6207618-9fa1-47b6-8ac0-3375f6139779
    2022-10-17 17:30:51,053 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: invoking result_received_cb ...
    2022-10-17 17:30:51,054 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-8 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:51,054 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Aggregation_weight missing for site-8 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:51,054 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: Contribution from site-8 ACCEPTED by the aggregator.
    2022-10-17 17:30:51,054 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:51,055 - Communicator - INFO - Received comments: simulator_server Received from site-8 (1140 Bytes, 1666042251 seconds). SubmitUpdate time: 0.0706486701965332 seconds
    2022-10-17 17:30:51,056 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=c6207618-9fa1-47b6-8ac0-3375f6139779]: result sent to server for task: name=train, id=c6207618-9fa1-47b6-8ac0-3375f6139779
    2022-10-17 17:30:51,056 - ClientTaskWorker - INFO - Finished one task run for client: site-8
    2022-10-17 17:30:51,056 - SimulatorClientRunner - INFO - Simulate Run client: site-1
    2022-10-17 17:30:51,056 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-8 
    E1017 17:30:52.058501552   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:53,551 - ClientRunner - INFO - [identity=site-1, run=simulate_job]: client runner started
    2022-10-17 17:30:53,551 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-1
    2022-10-17 17:30:53,623 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: assigned task to client site-1: name=train, id=a90507df-8de7-457a-b832-9b16c6758880
    2022-10-17 17:30:53,623 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: sent task assignment to client
    2022-10-17 17:30:53,624 - SimulatorServer - INFO - GetTask: Return task: train to client: site-1 (529ce6b4-5d71-4fe5-b6fc-ed9d14d26936) 
    2022-10-17 17:30:53,624 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.07228612899780273 seconds
    2022-10-17 17:30:53,625 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:53,626 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=a90507df-8de7-457a-b832-9b16c6758880
    2022-10-17 17:30:53,626 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:53,626 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Task name: train
    2022-10-17 17:30:53,627 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:53,627 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Model: 
    {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:53,627 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Current Round: 1
    2022-10-17 17:30:53,627 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Total Rounds: 3
    2022-10-17 17:30:53,627 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Client identity: site-1
    2022-10-17 17:30:53,628 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:53,628 - NPTrainer - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Model after training: {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:30:53,628 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: finished processing task
    2022-10-17 17:30:53,629 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:53,630 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:53,697 - SimulatorServer - INFO - received update from simulator_server_site-1_0 (1140 Bytes, 1666042253 seconds)
    2022-10-17 17:30:53,698 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job]: got result from client site-1 for task: name=train, id=a90507df-8de7-457a-b832-9b16c6758880
    2022-10-17 17:30:53,698 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: invoking result_received_cb ...
    2022-10-17 17:30:53,698 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-1 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:53,698 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Aggregation_weight missing for site-1 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:53,699 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: Contribution from site-1 ACCEPTED by the aggregator.
    2022-10-17 17:30:53,699 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:53,699 - Communicator - INFO - Received comments: simulator_server Received from site-1 (1140 Bytes, 1666042253 seconds). SubmitUpdate time: 0.0695347785949707 seconds
    2022-10-17 17:30:53,700 - ClientRunner - INFO - [identity=site-1, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a90507df-8de7-457a-b832-9b16c6758880]: result sent to server for task: name=train, id=a90507df-8de7-457a-b832-9b16c6758880
    2022-10-17 17:30:53,700 - ClientTaskWorker - INFO - Finished one task run for client: site-1
    2022-10-17 17:30:53,700 - SimulatorClientRunner - INFO - Simulate Run client: site-2
    2022-10-17 17:30:53,701 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-1 
    E1017 17:30:54.702952317   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:30:56,218 - ClientRunner - INFO - [identity=site-2, run=simulate_job]: client runner started
    2022-10-17 17:30:56,218 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-2
    2022-10-17 17:30:56,282 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: assigned task to client site-2: name=train, id=3df70456-7fbe-471c-a7a4-fac4a97fea04
    2022-10-17 17:30:56,282 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: sent task assignment to client
    2022-10-17 17:30:56,283 - SimulatorServer - INFO - GetTask: Return task: train to client: site-2 (3d9420db-1aa0-4142-adbb-2d8fc87a8e8b) 
    2022-10-17 17:30:56,283 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.06456303596496582 seconds
    2022-10-17 17:30:56,284 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:56,285 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=3df70456-7fbe-471c-a7a4-fac4a97fea04
    2022-10-17 17:30:56,285 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:56,286 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Task name: train
    2022-10-17 17:30:56,286 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:56,286 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Model: 
    {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:56,286 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Current Round: 1
    2022-10-17 17:30:56,286 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Total Rounds: 3
    2022-10-17 17:30:56,286 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Client identity: site-2
    2022-10-17 17:30:56,287 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:56,287 - NPTrainer - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Model after training: {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:30:56,287 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: finished processing task
    2022-10-17 17:30:56,288 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:56,289 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:56,367 - SimulatorServer - INFO - received update from simulator_server_site-2_0 (1140 Bytes, 1666042256 seconds)
    2022-10-17 17:30:56,367 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job]: got result from client site-2 for task: name=train, id=3df70456-7fbe-471c-a7a4-fac4a97fea04
    2022-10-17 17:30:56,368 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: invoking result_received_cb ...
    2022-10-17 17:30:56,368 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-2 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:56,368 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Aggregation_weight missing for site-2 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:56,368 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: Contribution from site-2 ACCEPTED by the aggregator.
    2022-10-17 17:30:56,368 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:56,369 - Communicator - INFO - Received comments: simulator_server Received from site-2 (1140 Bytes, 1666042256 seconds). SubmitUpdate time: 0.07973599433898926 seconds
    2022-10-17 17:30:56,370 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=3df70456-7fbe-471c-a7a4-fac4a97fea04]: result sent to server for task: name=train, id=3df70456-7fbe-471c-a7a4-fac4a97fea04
    2022-10-17 17:30:56,370 - ClientTaskWorker - INFO - Finished one task run for client: site-2
    2022-10-17 17:30:56,370 - SimulatorClientRunner - INFO - Simulate Run client: site-3
    2022-10-17 17:30:56,371 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-2 
    2022-10-17 17:30:58,889 - ClientRunner - INFO - [identity=site-3, run=simulate_job]: client runner started
    2022-10-17 17:30:58,890 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-3
    2022-10-17 17:30:58,981 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: assigned task to client site-3: name=train, id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2
    2022-10-17 17:30:58,981 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: sent task assignment to client
    2022-10-17 17:30:58,982 - SimulatorServer - INFO - GetTask: Return task: train to client: site-3 (738e9f46-877c-4856-bbd2-674eea8f5f27) 
    2022-10-17 17:30:58,982 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.09153056144714355 seconds
    2022-10-17 17:30:58,984 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:30:58,984 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2
    2022-10-17 17:30:58,985 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:30:58,985 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Task name: train
    2022-10-17 17:30:58,985 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Incoming data kind: WEIGHTS
    2022-10-17 17:30:58,985 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Model: 
    {'numpy_key': array([[ 2.,  3.,  4.],
        [ 5.,  6.,  7.],
        [ 8.,  9., 10.]], dtype=float32)}
    2022-10-17 17:30:58,985 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Current Round: 1
    2022-10-17 17:30:58,985 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Total Rounds: 3
    2022-10-17 17:30:58,985 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Client identity: site-3
    2022-10-17 17:30:58,986 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:30:58,986 - NPTrainer - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Model after training: {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:30:58,986 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: finished processing task
    2022-10-17 17:30:58,987 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:30:58,988 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:30:59,096 - SimulatorServer - INFO - received update from simulator_server_site-3_0 (1140 Bytes, 1666042259 seconds)
    2022-10-17 17:30:59,096 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job]: got result from client site-3 for task: name=train, id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2
    2022-10-17 17:30:59,097 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: invoking result_received_cb ...
    2022-10-17 17:30:59,097 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-3 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:30:59,097 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Aggregation_weight missing for site-3 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:30:59,098 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: Contribution from site-3 ACCEPTED by the aggregator.
    2022-10-17 17:30:59,098 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-3, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: finished processing client result by scatter_and_gather
    2022-10-17 17:30:59,098 - Communicator - INFO - Received comments: simulator_server Received from site-3 (1140 Bytes, 1666042259 seconds). SubmitUpdate time: 0.11000823974609375 seconds
    2022-10-17 17:30:59,099 - ClientRunner - INFO - [identity=site-3, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2]: result sent to server for task: name=train, id=a4deeb12-58f3-4489-a884-3a66a5d3f2a2
    2022-10-17 17:30:59,099 - ClientTaskWorker - INFO - Finished one task run for client: site-3
    2022-10-17 17:30:59,100 - SimulatorClientRunner - INFO - Simulate Run client: site-4
    2022-10-17 17:30:59,100 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-3 
    2022-10-17 17:31:01,290 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: task train exit with status TaskCompletionStatus.OK
    2022-10-17 17:31:01,290 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Start aggregation.
    2022-10-17 17:31:01,291 - DXOAggregator - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: aggregating 5 update(s) at round 1
    2022-10-17 17:31:01,291 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: End aggregation.
    2022-10-17 17:31:01,291 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Start persist model on server.
    2022-10-17 17:31:01,292 - NPModelPersistor - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/models/server.npy
    2022-10-17 17:31:01,292 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: End persist model on server.
    2022-10-17 17:31:01,292 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Round 1 finished.
    2022-10-17 17:31:01,292 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Round 2 started.
    2022-10-17 17:31:01,292 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: scheduled task train
    2022-10-17 17:31:01,619 - ClientRunner - INFO - [identity=site-4, run=simulate_job]: client runner started
    2022-10-17 17:31:01,619 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-4
    2022-10-17 17:31:01,689 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: assigned task to client site-4: name=train, id=acb068a0-93c5-46ff-a631-658366716991
    2022-10-17 17:31:01,690 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: sent task assignment to client
    2022-10-17 17:31:01,690 - SimulatorServer - INFO - GetTask: Return task: train to client: site-4 (2e9e56a9-ad05-48d0-bc60-9d322865f33e) 
    2022-10-17 17:31:01,690 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.06991314888000488 seconds
    2022-10-17 17:31:01,691 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:31:01,692 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=acb068a0-93c5-46ff-a631-658366716991
    2022-10-17 17:31:01,692 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:31:01,693 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Task name: train
    2022-10-17 17:31:01,693 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Incoming data kind: WEIGHTS
    2022-10-17 17:31:01,693 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Model: 
    {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:31:01,693 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Current Round: 2
    2022-10-17 17:31:01,693 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Total Rounds: 3
    2022-10-17 17:31:01,693 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Client identity: site-4
    2022-10-17 17:31:01,694 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:31:01,694 - NPTrainer - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Model after training: {'numpy_key': array([[ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]], dtype=float32)}
    2022-10-17 17:31:01,694 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: finished processing task
    2022-10-17 17:31:01,695 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:31:01,696 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:31:01,764 - SimulatorServer - INFO - received update from simulator_server_site-4_0 (1140 Bytes, 1666042261 seconds)
    2022-10-17 17:31:01,764 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job]: got result from client site-4 for task: name=train, id=acb068a0-93c5-46ff-a631-658366716991
    2022-10-17 17:31:01,764 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: invoking result_received_cb ...
    2022-10-17 17:31:01,765 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-4 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:31:01,765 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Aggregation_weight missing for site-4 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:31:01,765 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: Contribution from site-4 ACCEPTED by the aggregator.
    2022-10-17 17:31:01,765 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-4, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: finished processing client result by scatter_and_gather
    2022-10-17 17:31:01,766 - Communicator - INFO - Received comments: simulator_server Received from site-4 (1140 Bytes, 1666042261 seconds). SubmitUpdate time: 0.06962466239929199 seconds
    2022-10-17 17:31:01,766 - ClientRunner - INFO - [identity=site-4, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=acb068a0-93c5-46ff-a631-658366716991]: result sent to server for task: name=train, id=acb068a0-93c5-46ff-a631-658366716991
    2022-10-17 17:31:01,767 - ClientTaskWorker - INFO - Finished one task run for client: site-4
    2022-10-17 17:31:01,767 - SimulatorClientRunner - INFO - Simulate Run client: site-5
    2022-10-17 17:31:01,767 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-4 
    E1017 17:31:02.769286357   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:31:04,285 - ClientRunner - INFO - [identity=site-5, run=simulate_job]: client runner started
    2022-10-17 17:31:04,286 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-5
    2022-10-17 17:31:04,352 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: assigned task to client site-5: name=train, id=06119ad9-38af-461b-a8da-4eb9c0735927
    2022-10-17 17:31:04,353 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: sent task assignment to client
    2022-10-17 17:31:04,353 - SimulatorServer - INFO - GetTask: Return task: train to client: site-5 (7e822d77-7a7b-4ea4-9e67-b971416e456e) 
    2022-10-17 17:31:04,354 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.06682419776916504 seconds
    2022-10-17 17:31:04,355 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:31:04,355 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=06119ad9-38af-461b-a8da-4eb9c0735927
    2022-10-17 17:31:04,356 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:31:04,356 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Task name: train
    2022-10-17 17:31:04,356 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Incoming data kind: WEIGHTS
    2022-10-17 17:31:04,356 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Model: 
    {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:31:04,356 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Current Round: 2
    2022-10-17 17:31:04,356 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Total Rounds: 3
    2022-10-17 17:31:04,357 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Client identity: site-5
    2022-10-17 17:31:04,357 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:31:04,357 - NPTrainer - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Model after training: {'numpy_key': array([[ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]], dtype=float32)}
    2022-10-17 17:31:04,357 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: finished processing task
    2022-10-17 17:31:04,358 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:31:04,359 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:31:04,428 - SimulatorServer - INFO - received update from simulator_server_site-5_0 (1140 Bytes, 1666042264 seconds)
    2022-10-17 17:31:04,428 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job]: got result from client site-5 for task: name=train, id=06119ad9-38af-461b-a8da-4eb9c0735927
    2022-10-17 17:31:04,429 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: invoking result_received_cb ...
    2022-10-17 17:31:04,429 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-5 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:31:04,429 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Aggregation_weight missing for site-5 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:31:04,429 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: Contribution from site-5 ACCEPTED by the aggregator.
    2022-10-17 17:31:04,429 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-5, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: finished processing client result by scatter_and_gather
    2022-10-17 17:31:04,430 - Communicator - INFO - Received comments: simulator_server Received from site-5 (1140 Bytes, 1666042264 seconds). SubmitUpdate time: 0.07105779647827148 seconds
    2022-10-17 17:31:04,431 - ClientRunner - INFO - [identity=site-5, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=06119ad9-38af-461b-a8da-4eb9c0735927]: result sent to server for task: name=train, id=06119ad9-38af-461b-a8da-4eb9c0735927
    2022-10-17 17:31:04,431 - ClientTaskWorker - INFO - Finished one task run for client: site-5
    2022-10-17 17:31:04,431 - SimulatorClientRunner - INFO - Simulate Run client: site-6
    2022-10-17 17:31:04,432 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-5 
    E1017 17:31:05.433729198   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:31:06,955 - ClientRunner - INFO - [identity=site-6, run=simulate_job]: client runner started
    2022-10-17 17:31:06,955 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-6
    2022-10-17 17:31:07,038 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: assigned task to client site-6: name=train, id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6
    2022-10-17 17:31:07,038 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: sent task assignment to client
    2022-10-17 17:31:07,038 - SimulatorServer - INFO - GetTask: Return task: train to client: site-6 (0b291c05-0495-4936-aba8-69e735f03528) 
    2022-10-17 17:31:07,039 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.0822305679321289 seconds
    2022-10-17 17:31:07,040 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:31:07,041 - ClientRunner - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6
    2022-10-17 17:31:07,041 - ClientRunner - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:31:07,041 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Task name: train
    2022-10-17 17:31:07,041 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Incoming data kind: WEIGHTS
    2022-10-17 17:31:07,042 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Model: 
    {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:31:07,042 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Current Round: 2
    2022-10-17 17:31:07,042 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Total Rounds: 3
    2022-10-17 17:31:07,042 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Client identity: site-6
    2022-10-17 17:31:07,042 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:31:07,042 - NPTrainer - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Model after training: {'numpy_key': array([[ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]], dtype=float32)}
    2022-10-17 17:31:07,043 - ClientRunner - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: finished processing task
    2022-10-17 17:31:07,044 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:31:07,044 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:31:07,129 - SimulatorServer - INFO - received update from simulator_server_site-6_0 (1140 Bytes, 1666042267 seconds)
    2022-10-17 17:31:07,129 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job]: got result from client site-6 for task: name=train, id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6
    2022-10-17 17:31:07,130 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: invoking result_received_cb ...
    2022-10-17 17:31:07,130 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-6 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:31:07,130 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Aggregation_weight missing for site-6 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:31:07,130 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: Contribution from site-6 ACCEPTED by the aggregator.
    2022-10-17 17:31:07,131 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-6, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: finished processing client result by scatter_and_gather
    2022-10-17 17:31:07,131 - Communicator - INFO - Received comments: simulator_server Received from site-6 (1140 Bytes, 1666042267 seconds). SubmitUpdate time: 0.0865488052368164 seconds
    2022-10-17 17:31:07,132 - ClientRunner - INFO - [identity=site-6, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6]: result sent to server for task: name=train, id=cd6a86a6-4403-42c2-9d3b-efe3bc71e8d6
    2022-10-17 17:31:07,132 - ClientTaskWorker - INFO - Finished one task run for client: site-6
    2022-10-17 17:31:07,133 - SimulatorClientRunner - INFO - Simulate Run client: site-7
    2022-10-17 17:31:07,133 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-6 
    E1017 17:31:08.135085003   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:31:09,624 - ClientRunner - INFO - [identity=site-7, run=simulate_job]: client runner started
    2022-10-17 17:31:09,624 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-7
    2022-10-17 17:31:09,695 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: assigned task to client site-7: name=train, id=ef147e4d-3050-44f3-8495-2b2f77f85f17
    2022-10-17 17:31:09,695 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: sent task assignment to client
    2022-10-17 17:31:09,696 - SimulatorServer - INFO - GetTask: Return task: train to client: site-7 (e6127906-5283-45e3-b510-2866ff8a51a4) 
    2022-10-17 17:31:09,696 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.07119441032409668 seconds
    2022-10-17 17:31:09,697 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:31:09,698 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=ef147e4d-3050-44f3-8495-2b2f77f85f17
    2022-10-17 17:31:09,698 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:31:09,698 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Task name: train
    2022-10-17 17:31:09,699 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Incoming data kind: WEIGHTS
    2022-10-17 17:31:09,699 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Model: 
    {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:31:09,699 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Current Round: 2
    2022-10-17 17:31:09,699 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Total Rounds: 3
    2022-10-17 17:31:09,699 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Client identity: site-7
    2022-10-17 17:31:09,699 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:31:09,700 - NPTrainer - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Model after training: {'numpy_key': array([[ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]], dtype=float32)}
    2022-10-17 17:31:09,700 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: finished processing task
    2022-10-17 17:31:09,701 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:31:09,702 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:31:09,771 - SimulatorServer - INFO - received update from simulator_server_site-7_0 (1140 Bytes, 1666042269 seconds)
    2022-10-17 17:31:09,771 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job]: got result from client site-7 for task: name=train, id=ef147e4d-3050-44f3-8495-2b2f77f85f17
    2022-10-17 17:31:09,771 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: invoking result_received_cb ...
    2022-10-17 17:31:09,771 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-7 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:31:09,772 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Aggregation_weight missing for site-7 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:31:09,772 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: Contribution from site-7 ACCEPTED by the aggregator.
    2022-10-17 17:31:09,772 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-7, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: finished processing client result by scatter_and_gather
    2022-10-17 17:31:09,773 - Communicator - INFO - Received comments: simulator_server Received from site-7 (1140 Bytes, 1666042269 seconds). SubmitUpdate time: 0.07059240341186523 seconds
    2022-10-17 17:31:09,773 - ClientRunner - INFO - [identity=site-7, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=ef147e4d-3050-44f3-8495-2b2f77f85f17]: result sent to server for task: name=train, id=ef147e4d-3050-44f3-8495-2b2f77f85f17
    2022-10-17 17:31:09,774 - ClientTaskWorker - INFO - Finished one task run for client: site-7
    2022-10-17 17:31:09,774 - SimulatorClientRunner - INFO - Simulate Run client: site-8
    2022-10-17 17:31:09,774 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-7 
    2022-10-17 17:31:12,288 - ClientRunner - INFO - [identity=site-8, run=simulate_job]: client runner started
    2022-10-17 17:31:12,288 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-8
    2022-10-17 17:31:12,372 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: assigned task to client site-8: name=train, id=d4bd490d-de62-435f-95ac-451193707b2a
    2022-10-17 17:31:12,372 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: sent task assignment to client
    2022-10-17 17:31:12,372 - SimulatorServer - INFO - GetTask: Return task: train to client: site-8 (8fe14500-a6b8-47ad-b50c-aa22f9827830) 
    2022-10-17 17:31:12,373 - Communicator - INFO - Received from simulator_server server  (859 Bytes). getTask time: 0.08334660530090332 seconds
    2022-10-17 17:31:12,374 - FederatedClient - INFO - pull_task completed. Task name:train Status:True 
    2022-10-17 17:31:12,375 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: got task assignment: name=train, id=d4bd490d-de62-435f-95ac-451193707b2a
    2022-10-17 17:31:12,375 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: invoking task executor <class 'nvflare.app_common.np.np_trainer.NPTrainer'>
    2022-10-17 17:31:12,375 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Task name: train
    2022-10-17 17:31:12,375 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Incoming data kind: WEIGHTS
    2022-10-17 17:31:12,376 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Model: 
    {'numpy_key': array([[ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]], dtype=float32)}
    2022-10-17 17:31:12,376 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Current Round: 2
    2022-10-17 17:31:12,376 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Total Rounds: 3
    2022-10-17 17:31:12,376 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Client identity: site-8
    2022-10-17 17:31:12,376 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/model/best_numpy.npy
    2022-10-17 17:31:12,377 - NPTrainer - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Model after training: {'numpy_key': array([[ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]], dtype=float32)}
    2022-10-17 17:31:12,377 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: finished processing task
    2022-10-17 17:31:12,378 - FederatedClient - INFO - Starting to push execute result.
    2022-10-17 17:31:12,379 - Communicator - INFO - Send submitUpdate to simulator_server server
    2022-10-17 17:31:12,446 - SimulatorServer - INFO - received update from simulator_server_site-8_0 (1140 Bytes, 1666042272 seconds)
    2022-10-17 17:31:12,446 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job]: got result from client site-8 for task: name=train, id=d4bd490d-de62-435f-95ac-451193707b2a
    2022-10-17 17:31:12,447 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: invoking result_received_cb ...
    2022-10-17 17:31:12,447 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: NUM_STEPS_CURRENT_ROUND missing in meta of DXO from site-8 and set to default value, 1.0.  This kind of message will show 10 times at most.
    2022-10-17 17:31:12,447 - DXOAggregator - WARNING - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Aggregation_weight missing for site-8 and set to default value, 1.0 This kind of message will show 10 times at most.
    2022-10-17 17:31:12,447 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: Contribution from site-8 ACCEPTED by the aggregator.
    2022-10-17 17:31:12,447 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-8, peer_run=simulate_job, peer_rc=OK, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: finished processing client result by scatter_and_gather
    2022-10-17 17:31:12,448 - Communicator - INFO - Received comments: simulator_server Received from site-8 (1140 Bytes, 1666042272 seconds). SubmitUpdate time: 0.06935739517211914 seconds
    2022-10-17 17:31:12,449 - ClientRunner - INFO - [identity=site-8, run=simulate_job, peer=simulator_server, peer_run=simulate_job, task_name=train, task_id=d4bd490d-de62-435f-95ac-451193707b2a]: result sent to server for task: name=train, id=d4bd490d-de62-435f-95ac-451193707b2a
    2022-10-17 17:31:12,449 - ClientTaskWorker - INFO - Finished one task run for client: site-8
    2022-10-17 17:31:12,449 - SimulatorClientRunner - INFO - Simulate Run client: site-1
    2022-10-17 17:31:12,450 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-8 
    E1017 17:31:13.451732692   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:31:14,806 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: task train exit with status TaskCompletionStatus.OK
    2022-10-17 17:31:14,807 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Start aggregation.
    2022-10-17 17:31:14,807 - DXOAggregator - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: aggregating 5 update(s) at round 2
    2022-10-17 17:31:14,807 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: End aggregation.
    2022-10-17 17:31:14,808 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Start persist model on server.
    2022-10-17 17:31:14,808 - NPModelPersistor - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Saved numpy model to: /tmp/nvflare/workspace_folder/simulate_job/models/server.npy
    2022-10-17 17:31:14,808 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: End persist model on server.
    2022-10-17 17:31:14,808 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Round 2 finished.
    2022-10-17 17:31:14,808 - ScatterAndGather - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Finished ScatterAndGather Training.
    2022-10-17 17:31:14,809 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Workflow: scatter_and_gather finalizing ...
    2022-10-17 17:31:14,941 - ClientRunner - INFO - [identity=site-1, run=simulate_job]: client runner started
    2022-10-17 17:31:14,941 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-1
    2022-10-17 17:31:15,227 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-1, peer_run=simulate_job]: no current workflow - asked client to try again later
    2022-10-17 17:31:15,229 - ClientTaskWorker - INFO - Finished one task run for client: site-1
    2022-10-17 17:31:15,229 - SimulatorClientRunner - INFO - Simulate Run client: site-2
    2022-10-17 17:31:15,229 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-1 
    E1017 17:31:16.231305453   21930 fork_posix.cc:76]           Other threads are currently calling into gRPC, skipping fork() handlers
    2022-10-17 17:31:17,309 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: ABOUT_TO_END_RUN fired
    2022-10-17 17:31:17,729 - ClientRunner - INFO - [identity=site-2, run=simulate_job]: client runner started
    2022-10-17 17:31:17,729 - ClientTaskWorker - INFO - Initialize ClientRunner for client: site-2
    2022-10-17 17:31:17,795 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather, peer=site-2, peer_run=simulate_job]: server runner is finalizing - asked client to end the run
    2022-10-17 17:31:17,795 - SimulatorServer - INFO - GetTask: Return task: __end_run__ to client: site-2 (3d9420db-1aa0-4142-adbb-2d8fc87a8e8b) 
    2022-10-17 17:31:17,796 - Communicator - INFO - Received from simulator_server server  (348 Bytes). getTask time: 0.06571817398071289 seconds
    2022-10-17 17:31:17,797 - FederatedClient - INFO - pull_task completed. Task name:__end_run__ Status:True 
    2022-10-17 17:31:17,797 - ClientRunner - INFO - [identity=site-2, run=simulate_job, peer=simulator_server, peer_run=simulate_job]: server asked to end the run
    2022-10-17 17:31:17,797 - ClientTaskWorker - INFO - Finished one task run for client: site-2
    2022-10-17 17:31:17,797 - ClientTaskWorker - INFO - End the Simulator run.
    2022-10-17 17:31:17,797 - SimulatorClientRunner - INFO - Simulate Run client: site-3
    2022-10-17 17:31:17,798 - ClientTaskWorker - INFO - Clean up ClientRunner for : site-2 
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-1
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-2
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-3
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-4
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-5
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-6
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-7
    2022-10-17 17:31:17,798 - FederatedClient - INFO - Shutting down client: site-8
    2022-10-17 17:31:23,330 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: END_RUN fired
    2022-10-17 17:31:23,331 - ServerRunner - INFO - [identity=simulator_server, run=simulate_job, wf=scatter_and_gather]: Server runner finished.
    2022-10-17 17:31:26,301 - SimulatorServer - INFO - Server app stopped.
    
    
    2022-10-17 17:31:26,302 - SimulatorServer - INFO - shutting down server
    2022-10-17 17:31:26,302 - SimulatorServer - INFO - canceling sync locks
    2022-10-17 17:31:26,302 - SimulatorServer - INFO - server off

.. raw:: html

   </details>
   <br />

Run an NVFlare job
===================

This command will run the job following the meta.json in the job. The executing client list can be provided in the command line with the ``-c`` option
("client0,client1,client2,client3"). If there is any client not defined in the deploy_map of the meta.json, the simulator will report an error and not run.

.. code-block:: python

    nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -c client0,client1,client2,client3 -t 1

Note that the ``-n`` option is used to specify the number of clients like in the previous section above, but it is checked only if the ``-c`` option is not used.
The with the ``-n`` option, clients are automatically created up to the number provided after ``-n``, and they are named site-1, site-2, site-3, etc.

The output should be similar to above but with only four clients.

Run a job with no client name list
===================================

If there is no client name list provided and no number of clients (-n) option provided, the simulator extracts the list of client names from the deployment_map
in meta.json to run.

.. code-block:: python

    nvflare simulator NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag -w /tmp/nvflare/workspace_folder/ -t 1


.. note::

    The client name list option is used in priority over the number of clients option. When it's provided, it will be used as the simulated client name list.

**************************
Debug NVFlare Application
**************************

One of the goals for the Simulator is to enable researchers easily debug the NVFlare application. The FL simulator is implemented in a way of API design.
Actually, the Simulator application is also implemented using the Simulator API. The researchers can simply write a "main" python script like the Simulator
App, then place the script into their familiar Python IDE, add the NVFlare app into the python source codes path, then add the breakpoints to debug the
application run.

.. code-block:: python

    import argparse
    import sys
    from sys import platform

    from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


    def define_simulator_parser(simulator_parser):
        simulator_parser.add_argument("job_folder")
        simulator_parser.add_argument("-w", "--workspace", type=str, help="WORKSPACE folder")
        simulator_parser.add_argument("-n", "--n_clients", type=int, help="number of clients")
        simulator_parser.add_argument("-c", "--clients", type=str, help="client names list")
        simulator_parser.add_argument("-t", "--threads", type=int, help="number of parallel running clients")
        simulator_parser.add_argument("-gpu", "--gpu", type=str, help="list of GPU Device Ids, comma separated")
        simulator_parser.add_argument("-m", "--max_clients", type=int, default=100, help="max number of clients")


    def run_simulator(simulator_args):
        simulator = SimulatorRunner(
            job_folder=simulator_args.job_folder,
            workspace=simulator_args.workspace,
            clients=simulator_args.clients,
            n_clients=simulator_args.n_clients,
            threads=simulator_args.threads,
            gpu=simulator_args.gpu,
            max_clients=simulator_args.max_clients,
        )
        run_status = simulator.run()

        return run_status


    if __name__ == "__main__":
        """
        This is the main program when running the NVFlare Simulator. Use the Flare simulator API,
        create the SimulatorRunner object, do a setup(), then calls the run().
        """

        if sys.version_info < (3, 7):
            raise RuntimeError("Please use Python 3.7 or above.")

        parser = argparse.ArgumentParser()
        define_simulator_parser(parser)
        args = parser.parse_args()
        status = run_simulator(args)
        sys.exit(status)

******************************
Processes, Clients, and Events
******************************

Specifying number of processes
==============================
The simulator ``-t`` option provides the ability to specify how many processes to run the simulator with.

.. note::

    The ``-t`` and ``--threads`` option for simulator was originally due to clients running in separate threads.
    However each client now actually runs in a separate process. This distinction will not affect the user experience.

- N = number of clients (``-n``)
- T = number of processes (``-t``)

When running the simulator with fewer processes than clients (T < N)
the simulator will need to swap-in/out the clients for the processes, resulting in some of the clients running sequentially as processes are available.
This also will cause the ClientRunner/learner objects to go through setup and teardown in every round.
Using T < N is only needed when trying to simulate of large number of clients using a single machine with limited resources.

In most cases, run the simulator with the same number of processes as clients (T = N). The simulator will run the number of clients in separate processes at the same time. Each
client will always be running in memory with no swap-in/out, but it will require more resources available.

For the dataset / tensorboard initialization, you could make use of EventType.SWAP_IN and EventType.SWAP_OUT
in the application.

SWAP_IN and SWAP_OUT events
===========================
During FLARE simulator execution, the client Apps are executed in turn in the same execution thread. Each executing client App will go
fetch the task from the controller on the server, execute the task, and then submit the task results to the controller. Once finished submitting
results, the current client App will yield the executing thread to the next client App to execute.

If the client App needs to preserve some states for the next "execution turn" to continue, the client executor can make use of the ``SWAP_OUT``
event fired by the simulator engine to save the current states. When the client App gets the turn to execute again, use the ``SWAP_IN``
event to recover the previous saved states.

Multi-GPU and Separate Client Process with Simulator
====================================================
The simulator runs within the same process, and it will make use of a single GPU if it is detected with ``nvidia-smi``.
If there are multiple GPUs available and you want to make use of them all for the simulator run, you can use the
``-gpu`` option for this. The ``-gpu`` option provides the list of GPUs for the simulator to run on. The
clients list will be distributed among the GPUs.

For example: 

.. code-block::shell

    -c  c1,c2,c3,c4,c5 -gpu 0,1

The clients c1, c3, and c5 will run on GPU 0 in one process, and clients c2 and c4 will run on GPU 1 in another process.

The GPU numbers do not have to be unique. If you use ``-gpu 0,0``, this will run 2 separate client processes on GPU 0, assuming this GPU will have
enough memory to support the applications.

.. note::

    If you have invalid GPU IDs assigned and ``nvidia-smi`` is available, the simuilation will be aborted. Otherwise if ``nvidia-smi`` is not available,
    the simulation will run on CPU.

To change the MAX_CLIENTS
=========================
By default, the simulator runs with a maximum number of 100 clients. If you need to simulate a larger number of clients, use the "-m MAX_CLIENTS" option
to set the number of clients to run. The simulator can support more than 1000 clients with one run. You just need to make sure that the machine that the
simulator is running on has enough resources to support the parallel execution of the number of clients set.
