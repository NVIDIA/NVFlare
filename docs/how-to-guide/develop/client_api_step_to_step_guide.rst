.. _dl_to_fl_guide:

###########################
How to convert DL/ML to FL
###########################

We are using going to take deep learning code for example, other traditional ML examples
can be found in Hello-world example series. Assuming the dataset is ready to be used for training, converting an existing
stand-alone, centralized deep learning script to federated learning with NVFLARE is essentially involved the following few
steps

- Step 1. We need to decide what type training workflow: round robin (cyclic weight transfer), or scatter and gather type (weighted
average). For example, we can use federated weighted average: FedAvg. This essentially determine what type of server side
code to run.

- Step 2, we need convert the training scripts into client side local training scripts, so that it can receive global model,
perform the local training, and send the newly updated the local model back to the server.

- Step 3, we need to connect the server aggregation algorithm (FedAvg) and Client training code together to perform the
federated learning job.


NVIDIA FLARE provide a set of API stacks for data scientists to perform above steps.

-- Step 1, Data Scientists can use of the existing federated learning algorithms such as FedAvg. If you preferred to develop
customized algorithm, FLARE is offering new Collab API (will release soon) to simplified the development

-- Step 2, You can use FLARE Client API to convert the DL to FL with just a few lines of code changes ( see details below)

-- Step 3, We have Job Recipe API that connect the step 1 and 2. For example, FedAvgRecipe ( see details below).

Here are some code snippets from hello-pt, you can see the complete code and description in hello-pt example.


Client Code -- Client API
-------------------------

On the client side, the training workflow is as follows:

1. Receive the model from the FL server.
2. Perform local training on the received global model and/or evaluate the received global model for model selection.
3. Send the new model back to the FL server.

The client code (`client.py <./client.py>`_) is responsible for implementing this training workflow. Notice the training code is almost identical to a standard training PyTorch code.
The only difference is that we added a few lines to receive and send data to the server.

Using NVFlare's client API, we can easily adapt machine learning code that was written for centralized training and apply it in a federated scenario.
For a general use case, there are three essential methods to achieve this using the Client API :

- ``init()``: Initializes NVFlare Client API environment.
- ``receive()``: Receives model from the FL server.
- ``send()``: Sends the model to the FL server.

With these simple methods, the developers can use the Client API
to change their centralized training code to an FL scenario with
five lines of code changes as shown below.

.. code-block:: python

   import nvflare.client as flare

   flare.init() # 1. Initializes NVFlare Client API environment.
   input_model = flare.receive() # 2. Receives model from the FL server.
   params = input_model.params # 3. Obtain the required information from the received model.

   # original local training code
   new_params = local_train(params)

   output_model = flare.FLModel(params=new_params) # 4. Put the results in a new `FLModel`
   flare.send(output_model) # 5. Sends the model to the FL server.


Job Recipe
-----------

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

The JobRecipe code is saved to ``job.py``

Run Job
-------

From terminal simply run the job script to execute the job in a simulation environment.

.. code-block:: bash

   python job.py

See the complete example in ```hello-world/hello-pt/index.rst```
If you want to know more about the Client API, you can find the reference at `ref:client_api``;
If you want to know more about the JobRecipe API, you can find the reference at `ref:fed_job_api``;


