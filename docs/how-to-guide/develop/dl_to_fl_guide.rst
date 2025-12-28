.. _dl_to_fl_guide:

##################################################
How to Convert Deep Learning to Federated Learning
##################################################

This guide uses deep learning code as an example. Other traditional ML examples can be found in the
hello-world example series. Assuming the dataset is ready for training, converting an existing stand-alone,
centralized deep learning script to federated learning with NVIDIA FLARE involves the following steps:

- **Step 1**: Decide what type of training workflow to use: round-robin (cyclic weight transfer) or scatter-and-gather (weighted
  average). For example, we can use federated weighted averaging (FedAvg). This determines what server-side
  code to run.

- **Step 2**: Convert the training scripts into client-side local training scripts so that they can receive a global model,
  perform local training, and send the newly updated local model back to the server.

- **Step 3**: Connect the server aggregation algorithm (FedAvg) and client training code together to perform the
  federated learning job.


NVIDIA FLARE provides a set of API stacks for data scientists to perform the above steps:

- **Step 1**: Data scientists can use existing federated learning algorithms such as FedAvg. For custom
  algorithms, FLARE offers the Collab API to simplify development (coming soon). The ModelController API
  is also available for advanced customization.

- **Step 2**: You can use the FLARE Client API to convert DL to FL with just a few lines of code changes.

- **Step 3**: We have the Job Recipe API that connects steps 1 and 2. For example, ``FedAvgRecipe``.

Here are some code snippets from the :ref:`hello_pt` example. See the complete code and description there.

Client Code with Client API
---------------------------

On the client side, the training workflow is:

1. Receive the model from the FL server
2. Perform local training on the received global model and/or evaluate it for model selection
3. Send the updated model back to the FL server

The client code (``client.py``) implements this workflow. The training code is almost identical to
standard PyTorch training code—the only difference is a few lines to receive and send data to the server.

Using NVIDIA FLARE's Client API, you can easily adapt centralized ML code for federated scenarios.
The three essential methods are:

- ``init()``: Initializes the NVIDIA FLARE Client API environment.
- ``receive()``: Receives a model from the FL server.
- ``send()``: Sends the model to the FL server.

With these simple methods, developers can use the Client API
to change their centralized training code to an FL scenario with
five lines of code changes, as shown below:

.. code-block:: python

   import nvflare.client as flare

   flare.init()                    # 1. Initialize FLARE Client API
   input_model = flare.receive()   # 2. Receive model from FL server
   params = input_model.params     # 3. Extract model parameters

   # Original local training code
   new_params = local_train(params)

   output_model = flare.FLModel(params=new_params)  # 4. Wrap results in FLModel
   flare.send(output_model)                         # 5. Send model to FL server


Job Recipe
----------

The Job Recipe connects the client training script with the built-in federated averaging algorithm:

.. code-block:: python

   from nvflare.job_config.fed_avg_recipe import FedAvgRecipe
   from nvflare.simulation import SimEnv

   recipe = FedAvgRecipe(
       name="hello-pt",
       min_clients=n_clients,
       num_rounds=num_rounds,
       initial_model=SimpleNetwork(),
       train_script="client.py",
       train_args=f"--batch_size {batch_size}",
   )

   env = SimEnv(num_clients=n_clients, num_threads=n_clients)
   recipe.execute(env=env)

Save this code to ``job.py``.

Run the Job
-----------

From the terminal, run the job script to execute in a simulation environment:

.. code-block:: bash

   python job.py


References
----------

- :ref:`hello_pt` - Complete example with full source code
- :ref:`client_api` - Detailed Client API documentation
- :ref:`job_recipe` - Job Recipe API documentation
