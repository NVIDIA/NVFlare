.. _dl_to_fl_guide:

##################################################
How to Convert Deep Learning to Federated Learning
##################################################

We are going to use deep learning code as an example. Other traditional ML examples
can be found in the Hello-World example series. Assuming the dataset is ready to be used for training, converting an existing
stand-alone, centralized deep learning script to federated learning with NVIDIA FLARE involves the following steps:

- **Step 1**: Decide what type of training workflow to use: round-robin (cyclic weight transfer) or scatter-and-gather (weighted
  average). For example, we can use federated weighted averaging (FedAvg). This determines what server-side
  code to run.

- **Step 2**: Convert the training scripts into client-side local training scripts so that they can receive a global model,
  perform local training, and send the newly updated local model back to the server.

- **Step 3**: Connect the server aggregation algorithm (FedAvg) and client training code together to perform the
  federated learning job.


NVIDIA FLARE provides a set of API stacks for data scientists to perform the above steps:

- **Step 1**: Data scientists can use existing federated learning algorithms such as FedAvg. If you prefer to develop
  a customized algorithm, FLARE offers the Collab API to simplify development (coming soon). The older ModelController API can also be used if you can't use the new Collab API.

- **Step 2**: You can use the FLARE Client API to convert DL to FL with just a few lines of code changes.

- **Step 3**: We have the Job Recipe API that connects steps 1 and 2. For example, ``FedAvgRecipe``.

Here are some code snippets from hello-pt. You can see the complete code and description in the hello-pt example.

Client Code -- Client API
-------------------------

On the client side, the training workflow is as follows:

1. Receive the model from the FL server.
2. Perform local training on the received global model and/or evaluate the received global model for model selection.
3. Send the new model back to the FL server.

The client code (``client.py``) is responsible for implementing this training workflow. Notice that the training code is almost identical to standard PyTorch training code.
The only difference is that we add a few lines to receive and send data to the server.

Using NVIDIA FLARE's Client API, we can easily adapt machine learning code that was written for centralized training and apply it in a federated scenario.
For a general use case, there are three essential methods to achieve this using the Client API:

- ``init()``: Initializes the NVIDIA FLARE Client API environment.
- ``receive()``: Receives a model from the FL server.
- ``send()``: Sends the model to the FL server.

With these simple methods, developers can use the Client API
to change their centralized training code to an FL scenario with
five lines of code changes, as shown below:

.. code-block:: python

   import nvflare.client as flare

   flare.init()  # 1. Initialize NVIDIA FLARE Client API environment.
   input_model = flare.receive()  # 2. Receive model from the FL server.
   params = input_model.params  # 3. Obtain required information from the received model.

   # Original local training code
   new_params = local_train(params)

   output_model = flare.FLModel(params=new_params)  # 4. Put results in a new FLModel.
   flare.send(output_model)  # 5. Send the model to the FL server.


Job Recipe
----------

The Job Recipe specifies the ``client.py`` and selects the built-in federated averaging algorithm.

.. code-block:: python

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

The Job Recipe code is saved to ``job.py``.

Run Job
-------

From the terminal, simply run the job script to execute the job in a simulation environment:

.. code-block:: bash

   python job.py

See the complete example in :ref:`hello_pt`.

If you want to know more about the Client API, see :ref:`client_api`.

If you want to know more about the Job Recipe API, see :ref:`job_recipe`.
