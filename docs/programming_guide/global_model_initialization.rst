Global Model Initialization Approaches
======================================
Unlike non-federated machine learning, there are multiple approaches to initialize a global model. In previous NVFLARE releases,
the model is always initialized at the FL server-side and then broadcast to all the FL clients. Clients will take this initial model and start training.
With the FLARE 2.3.0 release, we introduce a new model initialization approach: client-side model initialization.
Users can decide to use either approach depending on their use cases and requirements. 

The benefits of the server-side model initialization allows the model to only initialize once in one place (server) and then be distributed to all clients,
so all clients have the same initial model. The potential issue with server-side model initialization might involve security concerns. For example, in order
to initialize a CNN model, we need to train the initial CNN model to generate the model. Running user-defined python code raises alarms to certain users. 

An alternative server-side model initialization without running python code is to have a predefined model file ready. This means users need to generate
a model outside and then manually upload the model file to a location that is accessible to the FL Server. This approach works but requires extra manual
steps outside of the training process. 

Client-side model initialization is an alternative approach that avoids server-side custom code as well as extra setup. Unlike the
server-side initialization approach, client-side initialization asks every client to send the initialized model as a pre-task in the workflow before
the training starts.  On the server side, once the server receives the initial models from the clients, the server can choose different strategies to leverage
the models from different clients: 

    - Select one model randomly from all clients' models, then use it as the global initial model
    - Apply some aggregation function to generate the global initial model 

In 2.3.0 release, a new InitializeGlobalWeights controller (:class:`nvflare.app_common.workflows.initialize_global_weights.InitializeGlobalWeights`) was implemented
to handle client-side model initialization. For details, see :ref:`initialize_global_weights_workflow`.

Here we have implemented the following strategies (specified with the "weight_method" argument):
    - If weight_method = "first", then use the weights reported from the first client;
    - If weight_method is "client", then only use the weights reported from the specified client.

If your use case demands a different strategy, then you can implement a new controller. 
